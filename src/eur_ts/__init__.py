from __future__ import annotations

import sys

import eis as _eis
import eis.artifacts as _artifacts
import eis.config as _config
import eis.data as _data
import eis.eval as _eval
import eis.train as _train
import eis.train.data as _train_data
import eis.train.interp as _train_interp
import eis.train.model as _train_model
import eis.train.runtime as _train_runtime
import eis.train.semantics as _train_semantics
import eis.train.training as _train_training

sys.modules[f"{__name__}.artifacts"] = _artifacts
sys.modules[f"{__name__}.config"] = _config
sys.modules[f"{__name__}.generator"] = _data
sys.modules[f"{__name__}.evaluator"] = _eval
sys.modules[f"{__name__}.trainer"] = _train
sys.modules[f"{__name__}.trainer.training"] = _train_training
sys.modules[f"{__name__}.trainer.data"] = _train_data
sys.modules[f"{__name__}.trainer.model"] = _train_model
sys.modules[f"{__name__}.trainer.interp"] = _train_interp
sys.modules[f"{__name__}.trainer.runtime"] = _train_runtime
sys.modules[f"{__name__}.trainer.semantics"] = _train_semantics

artifacts = _artifacts
config = _config
generator = _data
evaluator = _eval
trainer = _train

setattr(_eis, "artifacts", artifacts)
setattr(_eis, "config", config)
setattr(_eis, "generator", generator)
setattr(_eis, "evaluator", evaluator)
setattr(_eis, "trainer", trainer)

__all__: list[str] = []
