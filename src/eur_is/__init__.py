from __future__ import annotations

import sys

import eis.app as _app
import eis.app.backend as _backend
import eis.app.export as _export

sys.modules[f"{__name__}.backend"] = _backend
sys.modules[f"{__name__}.export"] = _export

backend = _backend
export = _export

setattr(_app, "backend", backend)
setattr(_app, "export", export)

__all__: list[str] = []
