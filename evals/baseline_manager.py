import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from app.config import get_settings


class BaselineComparisonResult(BaseModel):
    status: str  # "PASSED" or "REGRESSION_DETECTED"
    regressed_metrics: list[str]
    delta_faithfulness: float
    delta_context_precision: float
    current_metrics: dict[str, float]
    baseline_metrics: dict[str, float]


# Type alias matching Part 2 specification
RegressionReport = BaselineComparisonResult


class BaselineManager:
    """Manages baseline metrics and enforces continuous regression gates."""

    def __init__(
        self,
        baseline_file: str | None = None,
        regression_threshold: float | None = None,
    ):
        settings = get_settings()
        self.baseline_file = Path(baseline_file or "evals/baseline.json")
        self.regression_threshold = (
            regression_threshold
            if regression_threshold is not None
            else settings.REGRESSION_DELTA_THRESHOLD
        )

    def get_baseline(self) -> dict[str, Any]:
        if not self.baseline_file.exists():
            return {
                "faithfulness": 0.88,
                "context_precision": 0.85,
                "context_recall": 0.82,
                "answer_relevancy": 0.84,
                "created_at": "2026-03-31T00:00:00Z",
            }
        with open(self.baseline_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self, current: dict[str, float]) -> BaselineComparisonResult:
        baseline = self.get_baseline()

        delta_faith = baseline.get("faithfulness", 0.88) - current.get("faithfulness", 0.0)
        delta_prec = baseline.get("context_precision", 0.85) - current.get("context_precision", 0.0)

        regressed = []
        # Floating point tolerance check
        eps = 1e-6
        if delta_faith > (self.regression_threshold + eps):
            regressed.append(f"faithfulness (drop: {delta_faith:.4f} > {self.regression_threshold})")
        if delta_prec > (self.regression_threshold + eps):
            regressed.append(f"context_precision (drop: {delta_prec:.4f} > {self.regression_threshold})")

        status_str = "REGRESSION_DETECTED" if regressed else "PASSED"

        return BaselineComparisonResult(
            status=status_str,
            regressed_metrics=regressed,
            delta_faithfulness=round(delta_faith, 4),
            delta_context_precision=round(delta_prec, 4),
            current_metrics=current,
            baseline_metrics={k: v for k, v in baseline.items() if isinstance(v, (int, float))},
        )

    def check_regression(
        self,
        current_scores: dict[str, float],
        threshold: float = 0.05,
    ) -> BaselineComparisonResult:
        old_thresh = self.regression_threshold
        self.regression_threshold = threshold
        try:
            return self.compare(current_scores)
        finally:
            self.regression_threshold = old_thresh


    def update_baseline(self, new_metrics: dict[str, float]) -> None:
        payload = {
            "faithfulness": round(float(new_metrics.get("faithfulness", 0.88)), 4),
            "context_precision": round(float(new_metrics.get("context_precision", 0.85)), 4),
            "context_recall": round(float(new_metrics.get("context_recall", 0.82)), 4),
            "answer_relevancy": round(float(new_metrics.get("answer_relevancy", 0.84)), 4),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
