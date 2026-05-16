"""Compatibility shim: delegates to canonical CLI entrypoint."""

from eur_ts.trainer.main import main, parse_args, namespace_to_train_config

if __name__ == "__main__":
    main()

__all__ = ["main", "parse_args", "namespace_to_train_config"]
