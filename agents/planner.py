import sys
from core.client import get_client, MODEL

SYSTEM_PROMPT = """You are an ML project planner. Your job is to have a short conversation with the user to understand what machine learning experiment they want to run, then propose a concrete plan.

Gather this information:
1. What problem/dataset to use (suggest scikit-learn built-ins if the user is unsure: iris, wine, digits, breast_cancer, diabetes)
2. Task type (classification or regression)
3. Target accuracy/score (suggest 0.95 for classification, 0.80 R² for regression)
4. Number of optimization iterations (suggest 3–5)

Guidelines:
- Be concise. Ask only what you need.
- If the user gives enough info upfront, skip unnecessary questions.
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
                    "description": "Target accuracy or score for early stopping",
                },
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of the plan",
                },
            },
            "required": ["problem", "max_iterations", "target_score", "summary"],
        },
    }
]


def run() -> tuple[str, int, float]:
    """Interactive planning session. Returns (problem, max_iterations, target_score)."""
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
                return params["problem"], int(params["max_iterations"]), float(params["target_score"])
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
    print(f"  Max iterations: {params['max_iterations']}")
    print(f"  Target score:   {params['target_score']}")
    print("=" * 60)

    while True:
        answer = input("  実行する? [y/n]: ").strip().lower()
        if answer == "y":
            return True
        if answer == "n":
            return False
