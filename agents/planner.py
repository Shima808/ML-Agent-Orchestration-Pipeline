import sys
from core.client import get_client, MODEL

SYSTEM_PROMPT = """You are an ML project planner. Have a short conversation with the user to gather what they need, then propose a concrete plan.

Gather this information:
1. Problem description
2. Data source — scikit-learn built-in dataset OR CSV files (if CSV, ask for the directory path and target column name)
3. Metric to optimise — "accuracy" for classification, "auc" for binary competitions, "rmse" for regression
4. Target score (suggest: 0.95 for accuracy, 0.85 for AUC, 0.80 for R²/RMSE)
5. Number of iterations (suggest 3–5)
6. Submission file path (only if CSV competition mode — ask where to save the predictions)

Guidelines:
- Be concise. If the user gives enough info upfront, skip unnecessary questions.
- When you have all the information, call confirm_plan with the finalized parameters.
- Respond in the same language the user uses."""

TOOLS = [
    {
        "name": "confirm_plan",
        "description": "Propose the finalized plan to the user for confirmation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "Clear ML problem description passed to the Builder agent",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum optimization iterations",
                },
                "target_score": {
                    "type": "number",
                    "description": "Target score for early stopping",
                },
                "metric": {
                    "type": "string",
                    "description": "Metric key: 'accuracy', 'auc', 'rmse', etc.",
                },
                "data_dir": {
                    "type": "string",
                    "description": "Path to directory with train.csv / test.csv. Empty string if using sklearn built-ins.",
                },
                "target_col": {
                    "type": "string",
                    "description": "Target column name in the CSV. Empty string if using sklearn built-ins.",
                },
                "submission_path": {
                    "type": "string",
                    "description": "Path to save submission.csv. Empty string if not needed.",
                },
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of the plan",
                },
            },
            "required": [
                "problem", "max_iterations", "target_score",
                "metric", "data_dir", "target_col", "submission_path", "summary",
            ],
        },
    }
]


def run() -> dict:
    """Interactive planning session. Returns a dict with all plan parameters."""
    client = get_client()
    messages = []

    print("\n" + "=" * 60)
    print("  Planning Mode")
    print("=" * 60)
    print("  作りたいMLモデルについて教えてください。")
    print("  ('exit' で終了)\n")

    # Initial greeting from Claude
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=[{"role": "user", "content": "こんにちは。MLモデルを作りたいです。"}],
    )
    messages.append({"role": "user", "content": "こんにちは。MLモデルを作りたいです。"})
    messages.append({"role": "assistant", "content": response.content})

    _print_response(response)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[中断]")
            sys.exit(0)

        if user_input.lower() == "exit":
            sys.exit(0)
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        _print_response(response)

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use and tool_use.name == "confirm_plan":
            params = tool_use.input
            confirmed = _confirm_plan(params)
            if confirmed:
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "Plan confirmed."}],
                })
                return {
                    "problem":         params["problem"],
                    "max_iterations":  int(params["max_iterations"]),
                    "target_score":    float(params["target_score"]),
                    "metric":          params.get("metric", "accuracy") or "accuracy",
                    "data_dir":        params.get("data_dir") or None,
                    "target_col":      params.get("target_col") or None,
                    "submission_path": params.get("submission_path") or None,
                }
            else:
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "User wants to revise the plan."}],
                })
                print("\nPlanner: わかりました。どこを変えますか？\n")


def _print_response(response) -> None:
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text:
        print(f"\nPlanner: {text}\n")


def _confirm_plan(params: dict) -> bool:
    print("\n" + "=" * 60)
    print("  確認: 以下のプランで実行しますか？")
    print("=" * 60)
    print(f"  {params['summary']}")
    print(f"  Problem:        {params['problem']}")
    print(f"  Metric:         {params.get('metric', 'accuracy')}")
    print(f"  Max iterations: {params['max_iterations']}")
    print(f"  Target score:   {params['target_score']}")
    if params.get("data_dir"):
        print(f"  Data dir:       {params['data_dir']}")
        print(f"  Target col:     {params.get('target_col', '')}")
    if params.get("submission_path"):
        print(f"  Submission:     {params['submission_path']}")
    print("=" * 60)

    while True:
        answer = input("  実行する? [y/n]: ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
