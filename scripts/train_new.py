from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]

MODE_NEW = "new"
MODE_RESUME_AUTO = "resume-auto"
MODE_RESUME_EXPLICIT = "resume-explicit"


@dataclass(frozen=True)
class TrainOption:
    flag: str
    prompt: str
    default: str | None
    parser: Callable[[str], str]
    help_text: str
    is_boolean: bool = False


TRAIN_OPTIONS: tuple[TrainOption, ...] = (
    TrainOption("--data-dir", "Dataset directory", "data-1m", str, "Path to generated dataset."),
    TrainOption(
        "--output-dir",
        "Training output directory",
        "runs/arithmetic-small",
        str,
        "Checkpoint and log output directory.",
    ),
    TrainOption(
        "--resume-from",
        "Resume from checkpoint path",
        None,
        str,
        "Explicit checkpoint path to resume from.",
    ),
    TrainOption(
        "--resume",
        "Resume automatically from <output-dir>/checkpoint-last.pt",
        "n",
        lambda value: value.lower(),
        "Enable automatic resume.",
        is_boolean=True,
    ),
    TrainOption(
        "--additional-epochs",
        "Additional epochs after resume",
        None,
        lambda value: str(int(value)),
        "Only used when resuming.",
    ),
    TrainOption("--sequence-length", "Sequence length", "64", lambda value: str(int(value)), "Maximum context window."),
    TrainOption("--batch-size", "Batch size", "128", lambda value: str(int(value)), "Training batch size."),
    TrainOption("--epochs", "Epochs", "5", lambda value: str(int(value)), "Number of training epochs."),
    TrainOption("--learning-rate", "Learning rate", "3e-4", lambda value: str(float(value)), "AdamW learning rate."),
    TrainOption("--weight-decay", "Weight decay", "0.1", lambda value: str(float(value)), "AdamW weight decay."),
    TrainOption("--grad-clip", "Gradient clip", "1.0", lambda value: str(float(value)), "Gradient clipping norm."),
    TrainOption("--log-interval", "Log interval", "100", lambda value: str(int(value)), "Steps between log prints."),
    TrainOption("--eval-batches", "Validation batches per eval", "50", lambda value: str(int(value)), "Validation loss batches."),
    TrainOption(
        "--exact-match-samples",
        "Exact-match validation samples",
        "256",
        lambda value: str(int(value)),
        "Prompt samples for exact-match evaluation.",
    ),
    TrainOption("--max-new-tokens", "Max new tokens", "24", lambda value: str(int(value)), "Generation cap during validation."),
    TrainOption("--seed", "Seed", "42", lambda value: str(int(value)), "Random seed."),
    TrainOption("--device", "Device", "cuda", str, "Usually cuda, cpu, or auto."),
    TrainOption("--d-model", "Model width (d_model)", "256", lambda value: str(int(value)), "Embedding dimension."),
    TrainOption("--n-heads", "Attention heads", "4", lambda value: str(int(value)), "Number of attention heads."),
    TrainOption("--n-layers", "Transformer layers", "6", lambda value: str(int(value)), "Number of transformer blocks."),
    TrainOption("--mlp-hidden", "MLP hidden width", "1024", lambda value: str(int(value)), "MLP hidden dimension."),
    TrainOption("--dropout", "Dropout", "0.1", lambda value: str(float(value)), "Dropout rate."),
    TrainOption(
        "--checkpoint-keep-last",
        "Retain this many latest epoch checkpoints",
        "5",
        lambda value: str(int(value)),
        "Always keep this many latest physical checkpoints.",
    ),
    TrainOption(
        "--checkpoint-max-kept",
        "Maximum retained epoch checkpoints",
        "10",
        lambda value: str(int(value)),
        "<= 0 keeps all physical checkpoints.",
    ),
    TrainOption(
        "--checkpoint-keep-best",
        "Extra best checkpoints to keep",
        "1",
        lambda value: str(int(value)),
        "Retain extra best-performing checkpoints.",
    ),
    TrainOption(
        "--checkpoint-jump-threshold",
        "Exact-match jump threshold",
        "0.05",
        lambda value: str(float(value)),
        "Threshold for tagging comparison checkpoints.",
    ),
)

OPTION_BY_FLAG = {option.flag: option for option in TRAIN_OPTIONS}
COMMON_OPTION_FLAGS = (
    "--data-dir",
    "--output-dir",
    "--sequence-length",
    "--batch-size",
    "--epochs",
    "--learning-rate",
    "--weight-decay",
    "--grad-clip",
    "--log-interval",
    "--eval-batches",
    "--exact-match-samples",
    "--max-new-tokens",
    "--seed",
    "--device",
    "--d-model",
    "--n-heads",
    "--n-layers",
    "--mlp-hidden",
    "--dropout",
    "--checkpoint-keep-last",
    "--checkpoint-max-kept",
    "--checkpoint-keep-best",
    "--checkpoint-jump-threshold",
)


def prompt_line(message: str) -> str:
    try:
        return input(message)
    except EOFError as error:
        raise SystemExit("Interactive input ended unexpectedly.") from error


def prompt_value(option: TrainOption) -> str | bool | None:
    default_suffix = f" [{option.default}]" if option.default is not None else ""
    while True:
        raw = prompt_line(f"{option.prompt}{default_suffix}: ").strip()
        if not raw:
            if option.is_boolean:
                return (option.default or "n").lower() in {"y", "yes", "true", "1"}
            if option.default is None:
                return None
            raw = option.default
        if option.is_boolean:
            lowered = raw.lower()
            if lowered in {"y", "yes", "true", "1"}:
                return True
            if lowered in {"n", "no", "false", "0"}:
                return False
            print("Please answer y or n.")
            continue
        try:
            return option.parser(raw)
        except ValueError:
            print("Invalid value. Please try again.")


def prompt_training_mode() -> str:
    print("Training mode:")
    print("  1) New training run")
    print("  2) Resume from <output-dir>/checkpoint-last.pt")
    print("  3) Resume from an explicit checkpoint path")
    while True:
        raw = prompt_line("Choose a mode [1]: ").strip() or "1"
        if raw == "1":
            return MODE_NEW
        if raw == "2":
            return MODE_RESUME_AUTO
        if raw == "3":
            return MODE_RESUME_EXPLICIT
        print("Please choose 1, 2, or 3.")


def option_flags_for_mode(mode: str) -> tuple[str, ...]:
    flags = list(COMMON_OPTION_FLAGS)
    insert_at = 2
    if mode == MODE_RESUME_AUTO:
        flags.insert(insert_at, "--resume")
        flags.insert(insert_at + 1, "--additional-epochs")
    elif mode == MODE_RESUME_EXPLICIT:
        flags.insert(insert_at, "--resume-from")
        flags.insert(insert_at + 1, "--additional-epochs")
    return tuple(flags)


def collect_training_arguments() -> dict[str, str | bool | None]:
    print("Europa ALM-IS training launcher")
    print("Press Enter to accept defaults. Leave optional fields blank to skip them.")
    print()

    mode = prompt_training_mode()
    print()

    values: dict[str, str | bool | None] = {
        "--resume": mode == MODE_RESUME_AUTO,
        "--resume-from": None,
        "--additional-epochs": None,
    }
    for flag in option_flags_for_mode(mode):
        option = OPTION_BY_FLAG[flag]
        print(f"{option.flag}: {option.help_text}")
        values[option.flag] = prompt_value(option)
        print()

    resume_checkpoint = values.get("--resume-from")
    if resume_checkpoint:
        checkpoint_parent = Path(str(resume_checkpoint)).resolve().parent
        model_name_default = (
            checkpoint_parent.parent.name
            if checkpoint_parent.name == "checkpoints"
            else checkpoint_parent.name
        )
    else:
        model_name_default = Path(str(values["--output-dir"])).name
    model_name = prompt_line(f"Artifact model name [{model_name_default}]: ").strip() or model_name_default
    values["__model_name__"] = model_name
    return values


def build_train_command(values: dict[str, str | bool | None]) -> list[str]:
    command = ["uv", "run", "train", "train"]
    for option in TRAIN_OPTIONS:
        value = values[option.flag]
        if option.is_boolean:
            if value:
                command.append(option.flag)
            continue
        if value is None:
            continue
        command.extend([option.flag, str(value)])
    return command


def render_shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_and_capture(command: list[str], *, log_path: Path | None = None) -> int:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8", newline="")
    else:
        handle = None

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if handle is not None:
                handle.write(line)
        return process.wait()
    finally:
        if handle is not None:
            handle.close()


def write_train_script(script_path: Path, command: list[str]) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"{render_shell_command(command)}\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def main() -> None:
    values = collect_training_arguments()
    model_name = str(values.pop("__model_name__"))
    train_command = build_train_command(values)

    artifacts_dir = ROOT / "artifacts" / "models" / model_name
    train_script_path = artifacts_dir / "train.sh"
    evaluation_output_path = artifacts_dir / "evaluation.txt"
    output_dir = ROOT / str(values["--output-dir"])
    checkpoint_path = output_dir / "checkpoint-best.pt"

    print("Training command:")
    print(render_shell_command(train_command))
    print()

    confirm = prompt_line("Run this command? [Y/n]: ").strip().lower()
    if confirm in {"n", "no"}:
        raise SystemExit("Cancelled.")

    write_train_script(train_script_path, train_command)
    print(f"Saved replication script to {train_script_path.relative_to(ROOT)}")
    print()

    train_return_code = run_and_capture(train_command)
    if train_return_code != 0:
        raise SystemExit(train_return_code)

    if not checkpoint_path.exists():
        raise SystemExit(f"Training completed but no checkpoint was found at {checkpoint_path}.")

    evaluate_command = [
        "uv",
        "run",
        "evaluate",
        "--checkpoint",
        str(checkpoint_path),
        "--data-dir",
        str(values["--data-dir"]),
    ]

    print()
    print("Evaluation command:")
    print(render_shell_command(evaluate_command))
    print()

    evaluation_return_code = run_and_capture(evaluate_command, log_path=evaluation_output_path)
    if evaluation_return_code != 0:
        raise SystemExit(evaluation_return_code)

    print()
    print(f"Saved evaluation output to {evaluation_output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
