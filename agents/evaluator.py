import json
import subprocess
import sys
import textwrap
from core.cost_tracker import CostTracker

TIMEOUT_SECONDS = 30


def run(code: str, cost_tracker: CostTracker) -> dict:
    """Execute the generated code and return metrics dict."""
    print("\n[Evaluator] Running code...")

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        print(f"[Evaluator] Execution failed:\n{textwrap.indent(error_msg, '  ')}")
        cost_tracker.record("Evaluator", 0, 0)
        return {"accuracy": 0.0, "error": error_msg}

    stdout = result.stdout.strip()
    metrics = _parse_metrics(stdout)
    print(f"[Evaluator] Accuracy: {metrics.get('accuracy', 0.0):.4f}")
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
