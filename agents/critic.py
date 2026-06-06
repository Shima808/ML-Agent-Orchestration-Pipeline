from core.client import get_client, MODEL
from core.cost_tracker import CostTracker

SYSTEM_PROMPT = """You are an expert ML critic. Analyze the provided Python ML code and its evaluation metrics.
Provide specific, actionable improvement suggestions.

Focus on:
- Model selection (try different algorithms)
- Hyperparameter tuning (specific values to try)
- Feature engineering or preprocessing
- Ensemble methods if applicable

Be concise and specific. List 2-3 concrete improvements to try next."""


def run(
    code: str,
    metrics: dict,
    history_summary: str,
    cost_tracker: CostTracker,
) -> str:
    """Analyze code and metrics, return improvement suggestions."""
    client = get_client()

    accuracy = metrics.get("accuracy", 0.0)
    model_name = metrics.get("model", "unknown")

    user_content = (
        f"Current model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Code:\n```python\n{code}\n```"
    )
    if history_summary:
        user_content += f"\n\nPrevious iterations tried:\n{history_summary}"

    print("\n[Critic] Analyzing results...")
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    cost_tracker.record("Critic", response.usage.input_tokens, response.usage.output_tokens)

    critique = response.content[0].text.strip()
    print(f"[Critic] Suggestions:\n{critique}")
    return critique
