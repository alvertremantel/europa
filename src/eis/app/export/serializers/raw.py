"""Raw JSON serializers for export bundles."""

from __future__ import annotations

import json

from eis.app.backend.schemas import AnalyzeResponse, HealthResponse
from eis.app.export import layout


def build_raw_files(
    *, analysis: AnalyzeResponse, health: HealthResponse
) -> dict[str, bytes]:
    return {
        layout.RAW_ANALYZE_RESPONSE_PATH: json.dumps(
            analysis.model_dump(mode="json"), indent=2, sort_keys=True
        ).encode("utf-8"),
        layout.RAW_HEALTH_PATH: json.dumps(
            health.model_dump(mode="json"), indent=2, sort_keys=True
        ).encode("utf-8"),
    }
