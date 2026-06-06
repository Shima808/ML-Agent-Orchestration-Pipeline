from core.client import get_client, MODEL
from core.cost_tracker import CostTracker

SYSTEM_PROMPT = """You are an expert ML engineer specializing in scikit-learn.
Your job is to write Python code that trains and evaluates a machine learning model.

Rules:
- Use scikit-learn's Iris dataset (from sklearn.datasets import load_iris)
- Split data: test_size=0.2, random_state=42
- At the end, print ONLY a JSON object on the last line with this exact format:
  {"accuracy": <float>, "model": "<model name>"}
- Do NOT print anything else after the JSON line
- The code must be complete and runnable as-is (no user input required)
- Import all needed libraries at the top
"""


def run(problem: str, critique: str, cost_tracker: CostTracker) -> tuple[str, str]:
    """Generate sklearn ML code. Returns (code, approach_description)."""
    client = get_client()

    user_content = f"Problem: {problem}"
    if critique:
        user_content += f"\n\nPrevious model's weaknesses and improvements to apply:\n{critique}"
    else:
        user_content += "\n\nStart with a simple baseline model (e.g., LogisticRegression)."

    print("\n[Builder] Generating model code...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    cost_tracker.record("Builder", response.usage.input_tokens, response.usage.output_tokens)

    raw = response.content[0].text
    code, approach = _parse_response(raw)
    print(f"[Builder] Approach: {approach}")
    return code, approach


def _parse_response(text: str) -> tuple[str, str]:
    """Extract code block and approach from builder response."""
    lines = text.strip().split("\n")

    # Extract approach: first non-empty line before code block
    approach = ""
    code_lines = []
    in_code = False

    for line in lines:
        if line.startswith("```python"):
            in_code = True
            continue
        if line.startswith("```") and in_code:
            in_code = False
            continue
        if in_code:
            code_lines.append(line)
        elif not approach and line.strip() and not line.startswith("```"):
            approach = line.strip()

    if code_lines:
        return "\n".join(code_lines), approach or "ML model implementation"

    # Fallback: treat entire text as code if no code block found
    return text, approach or "ML model implementation"
