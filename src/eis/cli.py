from __future__ import annotations

import argparse
import os
import sys

import uvicorn

from .app.export.cli import main as export_main
from .config.cli import main as config_main
from .data.cli import main as data_main
from .eval.cli import main as eval_main
from .train.cli import main as train_main


def _wants_help(argv: list[str]) -> bool:
    return any(argument in {"-h", "--help"} for argument in argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Europa ALM-IS command surface."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Dataset generation commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_subparsers.add_parser("generate", help="Generate a dataset")

    config_parser = subparsers.add_parser("config", help="Training config commands")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", required=True
    )
    config_subparsers.add_parser("new", help="Create a new training config")
    config_subparsers.add_parser("guide", help="Show config guide")
    size_parser = config_subparsers.add_parser("size", help="Report model size")
    size_parser.add_argument("config", help="Path to a training TOML file")

    train_parser = subparsers.add_parser("train", help="Training and inference")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_run_parser = train_subparsers.add_parser(
        "run", help="Train a model from a TOML config file"
    )
    train_run_parser.add_argument("config", help="Path to a training TOML file")
    predict_parser = train_subparsers.add_parser(
        "predict", help="Generate an answer from a saved checkpoint"
    )
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--prompt", required=True)
    predict_parser.add_argument("--max-new-tokens", type=int, default=24)
    predict_parser.add_argument("--device", default="auto")

    eval_parser = subparsers.add_parser("eval", help="Checkpoint evaluation")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_subparsers.add_parser("run", help="Evaluate a checkpoint")
    eval_run.add_argument("remainder", nargs=argparse.REMAINDER)

    app_parser = subparsers.add_parser("app", help="Dashboard commands")
    app_subparsers = app_parser.add_subparsers(dest="app_command", required=True)
    serve_parser = app_subparsers.add_parser("serve", help="Run the backend API")
    serve_parser.add_argument("--checkpoint", type=str)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")

    export_parser = subparsers.add_parser("export", help="Export analysis bundles")
    export_parser.add_argument("remainder", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command = argv[0]
    if command == "data":
        rest = argv[1:]
        if _wants_help(rest):
            data_main(["--help"])
            return
        if len(argv) < 2:
            parser.parse_args(["data", "--help"])
            return
        if argv[1] != "generate":
            parser.parse_args(argv)
            return
        data_main(argv[2:])
        return
    if command == "config":
        rest = argv[1:]
        if _wants_help(rest):
            config_main(["--help"])
            return
        if len(argv) < 2:
            parser.parse_args(["config", "--help"])
            return
        if argv[1] == "new":
            config_main(["--new"])
        elif argv[1] == "guide":
            config_main(["--guide"])
        elif argv[1] == "size":
            if len(argv) < 3:
                raise SystemExit("eis config size requires a config path")
            config_main(["--size", argv[2]])
        else:
            parser.parse_args(argv)
        return
    if command == "train":
        rest = argv[1:]
        if len(rest) == 0:
            parser.parse_args(["train", "--help"])
            return
        if _wants_help(rest):
            if rest[0] in {"run", "predict"}:
                train_main(["train" if rest[0] == "run" else "predict", "--help"])
            else:
                train_main(["--help"])
            return
        if len(argv) < 2:
            parser.parse_args(argv)
            return
        if argv[1] == "run":
            train_main(["train", *argv[2:]])
        elif argv[1] == "predict":
            train_main(["predict", *argv[2:]])
        else:
            parser.parse_args(argv)
        return
    if command == "eval":
        rest = argv[1:]
        if _wants_help(rest):
            eval_main(["--help"])
            return
        if len(argv) < 2:
            parser.parse_args(["eval", "--help"])
            return
        if argv[1] != "run":
            parser.parse_args(argv)
            return
        eval_main(argv[2:])
        return
    if command == "export":
        if _wants_help(argv[1:]):
            export_main(["--help"])
            return
        export_main(argv[1:])
        return
    if command == "app":
        rest = argv[1:]
        if len(rest) == 0:
            parser.parse_args(["app", "--help"])
            return
        if _wants_help(rest):
            if rest[0] == "serve":
                app_parser = argparse.ArgumentParser(prog="eis app serve")
                app_parser.add_argument("--checkpoint", type=str)
                app_parser.add_argument("--host", default="127.0.0.1")
                app_parser.add_argument("--port", type=int, default=8000)
                app_parser.add_argument("--reload", action="store_true")
                app_parser.print_help()
            else:
                parser.parse_args(argv)
            return
        if len(argv) < 2 or argv[1] != "serve":
            parser.parse_args(argv)
            return
        app_parser = argparse.ArgumentParser(prog="eis app serve")
        app_parser.add_argument("--checkpoint", type=str)
        app_parser.add_argument("--host", default="127.0.0.1")
        app_parser.add_argument("--port", type=int, default=8000)
        app_parser.add_argument("--reload", action="store_true")
        app_args = app_parser.parse_args(argv[2:])
        if app_args.checkpoint:
            os.environ["EUR_IS_CHECKPOINT_PATH"] = app_args.checkpoint
        uvicorn.run(
            "eis.app.backend.main:app",
            host=app_args.host,
            port=app_args.port,
            reload=app_args.reload,
        )
        return

    raise SystemExit(f"unsupported command: {command}")
