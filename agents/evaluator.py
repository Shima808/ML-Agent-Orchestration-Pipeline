import json
import subprocess
import sys
import textwrap
from core.cost_tracker import CostTracker
from core.types import MLPipelineState

_DEFAULT_TIMEOUT = 30
_CSV_TIMEOUT = 180  # LightGBM / CatBoost on real data needs more time


def run(code: str, cost_tracker: CostTracker, state: MLPipelineState | None = None) -> dict:
    """Execute the generated code and return metrics dict."""
    metric_key = state.metric if state else "accuracy"
    timeout = _CSV_TIMEOUT if (state and state.data_dir) else _DEFAULT_TIMEOUT

    print("\n[Evaluator] Running code...")

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        print(f"[Evaluator] Execution failed:\n{textwrap.indent(error_msg, '  ')}")
        cost_tracker.record("Evaluator", 0, 0)
        return {metric_key: 0.0, "error": error_msg}

    stdout = result.stdout.strip()
    metrics = _parse_metrics(stdout)
    score = metrics.get(metric_key, 0.0)
    print(f"[Evaluator] {metric_key}: {score:.4f}")
    cost_tracker.record("Evaluator", 0, 0)
    return metrics


def _parse_metrics(stdout: str) -> dict:
    """Extract the last JSON line from stdout."""
    lines = stdout.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"accuracy": 0.0, "error": "No JSON metrics found in output"}
