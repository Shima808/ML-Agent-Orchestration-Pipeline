from core.client import get_client, MODEL
from core.cost_tracker import CostTracker
from core.types import MLPipelineState

_SKLEARN_SYSTEM_PROMPT = """You are an expert ML engineer specializing in scikit-learn.
Your job is to write Python code that trains and evaluates a machine learning model.

Rules:
- Use scikit-learn's built-in datasets (Iris, Wine, etc.) unless the problem specifies otherwise
- Split data: test_size=0.2, random_state=42
- At the end, print ONLY a JSON object on the last line with this exact format:
  {"accuracy": <float>, "model": "<model name>"}
- Do NOT print anything else after the JSON line
- The code must be complete and runnable as-is (no user input required)
- Import all needed libraries at the top
"""


def _csv_system_prompt(state: MLPipelineState) -> str:
    sub_line = ""
    if state.submission_path:
        sub_line = (
            f"- Save test-set predictions to '{state.submission_path}' "
            f"(CSV with 'Id' and '{state.target_col}' columns, probabilities 0-1)\n"
        )
    return f"""You are an expert ML engineer for tabular competitions.
Your job is to write Python code that trains a model on CSV data and reports cross-validation performance.

Data:
- Training data : '{state.data_dir}/train.csv'
- Test data     : '{state.data_dir}/test.csv'
- Target column : '{state.target_col}'

Rules:
- Load data with pandas; use 'Id' column as index if present (do not use it as a feature)
- Encode categorical columns with LabelEncoder or pd.get_dummies
- Impute missing values (median for numeric, mode for categorical)
- Use LightGBM as the primary model (preferred); scikit-learn is allowed as fallback
- Evaluate with StratifiedKFold (5 folds) and report mean CV {state.metric}
- At the end, print ONLY a JSON object on the last line:
  {{"{state.metric}": <float>, "model": "<model name>"}}
{sub_line}- Do NOT print anything else after the JSON line
- The code must be complete and runnable as-is
- Import all needed libraries at the top
"""


def run(state: MLPipelineState, critique: str, cost_tracker: CostTracker) -> tuple[str, str]:
    """Generate ML code. Returns (code, approach_description)."""
    client = get_client()

    system = _csv_system_prompt(state) if state.data_dir else _SKLEARN_SYSTEM_PROMPT

    user_content = f"Problem: {state.problem}"
    if critique:
        user_content += f"\n\nPrevious model's weaknesses and improvements to apply:\n{critique}"
    else:
        user_content += "\n\nStart with a simple baseline model."

    print("\n[Builder] Generating model code...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=system,
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

    return text, approach or "ML model implementation"
