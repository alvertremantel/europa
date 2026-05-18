from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

from torch import Tensor, nn
from torch.utils.data import DataLoader

from eur_ts.artifacts import toml_text
from eur_ts.config import TrainConfig
from ..curriculum import (
    build_balanced_example_sample,
    count_curriculum_groups,
    resample_for_curriculum,
)
from ..data import (
    ArithmeticExample,
    ExampleSequenceDataset,
    TokenBlockDataset,
    load_examples,
    load_token_stream_with_roles,
    transform_examples,
)
from ..inference import (
    evaluate_balanced_loss,
    evaluate_exact_match,
    evaluate_exact_match_examples,
    evaluate_loss,
    loss_for_batch,
    loss_for_example_batch,
)
from ..utils import (
    configure_runtime,
    device_metadata,
    parameter_count,
    resolve_device,
    set_seed,
)

from .checkpointing import (
    CheckpointManager,
    build_checkpoint_payload,
)
from .metadata import (
    resolve_target_epoch,
    scratchpad_fraction,
    write_history,
    write_run_metadata,
)
from .resume import initialize_training_state, resolve_resume_path
from .state import capture_rng_state


def train_model(config: TrainConfig) -> None:
    device = resolve_device(config.device)
    configure_runtime(device)
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(config, output_dir)
    if resume_path is None:
        set_seed(config.seed)

    (
        tokenizer,
        model,
        effective_model_config,
        optimizer,
        history,
        start_epoch,
        best_exact_match,
        global_step,
        resume_source,
        resumed_from_epoch,
    ) = initialize_training_state(
        config=config,
        device=device,
        resume_path=resume_path,
    )

    print(toml_text({"train_config": asdict(config)}).rstrip())
    print(
        toml_text(
            {
                "runtime": {
                    "parameters": parameter_count(model),
                    "vocab_size": tokenizer.vocab_size,
                    **device_metadata(device),
                }
            }
        ).rstrip()
    )

    val_tokens, val_position_ids = load_token_stream_with_roles(
        data_dir / "val.txt", tokenizer
    )
    val_dataset = TokenBlockDataset(
        val_tokens,
        val_position_ids,
        effective_model_config.sequence_length,
    )
    if len(val_dataset) == 0:
        raise ValueError(
            "validation dataset is too small for the configured sequence length"
        )

    train_examples: list[ArithmeticExample] | None = None
    static_train_loader: (
        DataLoader[tuple[Tensor, Tensor, Tensor]]
        | DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]
    )
    if config.training_mode == "token_stream":
        if config.training_format != "final_only":
            raise ValueError(
                'scratchpad training formats require training.training_mode = "examples"'
            )
        if config.curriculum_name is not None:
            raise ValueError(
                'curriculum presets require training.training_mode = "examples"'
            )
        train_tokens, train_position_ids = load_token_stream_with_roles(
            data_dir / "train.txt", tokenizer
        )
        train_dataset = TokenBlockDataset(
            train_tokens,
            train_position_ids,
            effective_model_config.sequence_length,
        )
        if len(train_dataset) == 0:
            raise ValueError(
                "training dataset is too small for the configured sequence length"
            )
        static_train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=device.type == "cuda",
        )
        if len(static_train_loader) == 0:
            raise ValueError(
                "training dataset produced no batches; lower optimization.batch_size"
            )
        print(f"loaded token-stream train blocks={len(train_dataset)}")
    elif config.training_mode == "examples":
        raw_examples = load_examples(data_dir / "train.txt", include_metadata=True)
        train_examples = transform_examples(
            raw_examples,
            training_format=config.training_format,
        )
        example_dataset = ExampleSequenceDataset(
            train_examples,
            tokenizer,
            effective_model_config.sequence_length,
            skip_overlong=config.skip_overlong_examples,
        )
        if example_dataset.skipped_by_format:
            print(
                toml_text(
                    {"skipped_overlong_examples": example_dataset.skipped_by_format}
                ).rstrip()
            )
        static_train_loader = DataLoader(
            example_dataset,
            batch_size=config.batch_size,
            shuffle=config.curriculum_name is None,
            drop_last=True,
            pin_memory=device.type == "cuda",
        )
        if len(static_train_loader) == 0:
            raise ValueError(
                "example training dataset produced no batches; lower optimization.batch_size"
            )
        print(
            toml_text(
                {
                    "training_data": {
                        "loaded_examples": len(example_dataset),
                        "curriculum_group_counts": count_curriculum_groups(
                            example_dataset.examples
                        ),
                    }
                }
            ).rstrip()
        )
    else:
        raise ValueError("training_mode must be token_stream or examples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    balanced_val_dataset: ExampleSequenceDataset | None = None
    balanced_val_examples: list[ArithmeticExample] | None = None
    if config.balanced_val_enabled:
        val_examples = transform_examples(
            load_examples(data_dir / "val.txt", include_metadata=True),
            training_format=config.training_format,
        )
        balanced_val_examples = build_balanced_example_sample(
            val_examples,
            group_by=config.balanced_val_group_by,
            sample_size_per_group=config.balanced_val_sample_size_per_group,
            seed=config.balanced_val_seed,
        )
        balanced_val_dataset = ExampleSequenceDataset(
            balanced_val_examples,
            tokenizer,
            effective_model_config.sequence_length,
            skip_overlong=config.skip_overlong_examples,
        )
        balanced_val_examples = balanced_val_dataset.examples
        print(
            toml_text(
                {
                    "balanced_validation": {
                        "balanced_val_examples": len(balanced_val_dataset),
                        "balanced_val_group_by": config.balanced_val_group_by,
                        "balanced_val_curriculum_group_counts": count_curriculum_groups(
                            balanced_val_dataset.examples
                        ),
                    }
                }
            ).rstrip()
        )

    checkpoint_manager = CheckpointManager(output_dir, config)
    run_metadata_path = output_dir / "run-metadata.toml"
    run_started_at = time.time()
    target_epoch = resolve_target_epoch(config, start_epoch - 1)
    if target_epoch < start_epoch - 1:
        raise ValueError(
            f"target epoch {target_epoch} is before checkpoint epoch {start_epoch - 1}"
        )
    if target_epoch == start_epoch - 1:
        print(
            f"No epochs remain to train; checkpoint is already at epoch {target_epoch}."
        )
        write_run_metadata(
            run_metadata_path,
            config=config,
            model_config=effective_model_config,
            model=model,
            device=device,
            resume_source=resume_source,
            run_started_at=run_started_at,
            run_completed_at=time.time(),
            history=history,
        )
        return

    for epoch in range(start_epoch, target_epoch + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss_total = 0.0
        epoch_batches = 0
        log_loss_total = 0.0
        log_batches = 0
        curriculum_stage: str | None = None
        curriculum_stage_index: int | None = None
        curriculum_sample_counts: dict[str, int] | None = None
        curriculum_sampling_weights: dict[str, float] | None = None

        train_loader = static_train_loader
        if config.training_mode == "examples" and config.curriculum_name is not None:
            assert train_examples is not None
            sampled_examples, counts, weights, stage_name, stage_index = (
                resample_for_curriculum(
                    train_examples,
                    curriculum_name=config.curriculum_name,
                    epoch=epoch,
                    seed=config.seed,
                )
            )
            curriculum_stage = stage_name
            curriculum_stage_index = stage_index
            curriculum_sample_counts = counts
            curriculum_sampling_weights = weights
            epoch_dataset = ExampleSequenceDataset(
                sampled_examples,
                tokenizer,
                effective_model_config.sequence_length,
                skip_overlong=config.skip_overlong_examples,
            )
            train_loader = DataLoader(
                epoch_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=True,
                pin_memory=device.type == "cuda",
            )
            if len(train_loader) == 0:
                raise ValueError(
                    "curriculum epoch produced no batches; lower optimization.batch_size"
                )
            print(
                toml_text(
                    {
                        "curriculum_epoch": {
                            "epoch": epoch,
                            "curriculum_stage": curriculum_stage,
                            "curriculum_stage_index": curriculum_stage_index,
                            "curriculum_sampling_weights": curriculum_sampling_weights,
                            "curriculum_sample_counts": curriculum_sample_counts,
                        }
                    }
                ).rstrip()
            )

        for step, batch in enumerate(train_loader, start=1):
            if config.training_mode == "examples":
                inputs, input_position_ids, targets, loss_mask = batch
            else:
                inputs, input_position_ids, targets = batch
                loss_mask = None
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            input_position_ids = input_position_ids.to(
                device, non_blocking=device.type == "cuda"
            )
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if loss_mask is not None:
                loss_mask = loss_mask.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            if loss_mask is None:
                loss = loss_for_batch(
                    model,
                    inputs,
                    targets,
                    position_ids=input_position_ids,
                )
            else:
                loss = loss_for_example_batch(
                    model,
                    inputs,
                    targets,
                    loss_mask,
                    position_ids=input_position_ids,
                ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss_total += step_loss
            log_loss_total += step_loss
            epoch_batches += 1
            log_batches += 1
            global_step += 1

            if step % config.log_interval == 0:
                average_loss = log_loss_total / max(log_batches, 1)
                print(
                    f"epoch={epoch} step={step} train_loss={average_loss:.4f} "
                    f"elapsed={time.perf_counter() - epoch_start:.1f}s"
                )
                log_loss_total = 0.0
                log_batches = 0

        train_loss = epoch_loss_total / max(epoch_batches, 1)
        val_loss = evaluate_loss(model, val_loader, device, config.eval_batches)
        balanced_val_loss = None
        balanced_exact_match = None
        if balanced_val_dataset is not None and balanced_val_examples is not None:
            balanced_val_loss = evaluate_balanced_loss(
                model,
                balanced_val_dataset,
                batch_size=config.balanced_val_batch_size or config.batch_size,
                device=device,
            )
            balanced_exact_match = evaluate_exact_match_examples(
                model=model,
                tokenizer=tokenizer,
                examples=balanced_val_examples,
                max_new_tokens=config.max_new_tokens,
                device=device,
            )
        exact_match = evaluate_exact_match(
            model=model,
            tokenizer=tokenizer,
            file_path=data_dir / "val.txt",
            sample_count=config.exact_match_samples,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
        best_exact_match = max(best_exact_match, exact_match)

        epoch_duration_seconds = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        metrics: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "exact_match": exact_match,
            "epoch_duration_seconds": epoch_duration_seconds,
            "learning_rate": lr,
            "checkpoint_path": None,
            "checkpoint_roles": [],
        }
        if balanced_val_loss is not None:
            metrics["balanced_val_loss"] = balanced_val_loss
        if balanced_exact_match is not None:
            metrics["balanced_exact_match"] = balanced_exact_match
        if curriculum_stage is not None:
            metrics["curriculum_stage"] = curriculum_stage
            metrics["curriculum_stage_index"] = curriculum_stage_index
            metrics["curriculum_sample_counts"] = curriculum_sample_counts
            metrics["curriculum_sampling_weights"] = curriculum_sampling_weights
        if config.training_mode == "examples":
            metrics["scratchpad_fraction"] = scratchpad_fraction(
                balanced_val_examples
                if balanced_val_examples is not None
                else train_examples or []
            )
        if resumed_from_epoch is not None:
            metrics["resumed_from"] = resumed_from_epoch

        payload = build_checkpoint_payload(
            model=model,
            tokenizer=tokenizer,
            train_config=config,
            epoch=epoch,
            val_loss=val_loss,
            exact_match=exact_match,
            optimizer_state=cast(dict[str, object], optimizer.state_dict()),
            rng_state=capture_rng_state(),
            history=[*history, dict(metrics)],
            best_exact_match=best_exact_match,
            global_step=global_step,
            checkpoint_roles=[],
            resume_source=resume_source,
            train_loss=train_loss,
        )
        checkpoint_path, checkpoint_roles = checkpoint_manager.save_epoch(
            payload=payload,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            exact_match=exact_match,
            global_step=global_step,
        )

        metrics["checkpoint_path"] = str(checkpoint_path.relative_to(output_dir))
        metrics["checkpoint_roles"] = checkpoint_roles
        history.append(metrics)
        print(toml_text({"epoch_metrics": metrics}).rstrip())

        write_history(output_dir / "history.toml", history)
        write_run_metadata(
            run_metadata_path,
            config=config,
            model_config=effective_model_config,
            model=model,
            device=device,
            resume_source=resume_source,
            run_started_at=run_started_at,
            run_completed_at=None,
            history=history,
        )

    write_run_metadata(
        run_metadata_path,
        config=config,
        model_config=effective_model_config,
        model=model,
        device=device,
        resume_source=resume_source,
        run_started_at=run_started_at,
        run_completed_at=time.time(),
        history=history,
    )
