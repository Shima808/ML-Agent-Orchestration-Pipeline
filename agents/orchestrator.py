import json
from core.client import get_client, MODEL
from core.types import MLPipelineState, IterationResult
from core.cost_tracker import CostTracker
from agents import builder, evaluator, critic

SYSTEM_PROMPT = """You are an ML pipeline orchestrator. You control a loop that iteratively improves a machine learning model.

You have access to these tools:
- call_builder: Generate or improve ML model code (optionally incorporating critique)
- call_evaluator: Execute the current code and get accuracy metrics
- call_critic: Analyze code + metrics and suggest improvements
- finish_pipeline: Stop the loop when satisfied or max iterations reached

Strategy:
1. Always start with call_builder to generate initial code
2. Then call_evaluator to get metrics
3. If accuracy < target and iterations remain, call_critic then call_builder again
4. Call finish_pipeline when done

Be decisive. One tool call per turn."""

TOOLS = [
    {
        "name": "call_builder",
        "description": "Generate or improve ML model code. Optionally pass critique to guide improvements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "use_critique": {
                    "type": "boolean",
                    "description": "Whether to include the latest critique in the builder prompt",
                }
            },
            "required": ["use_critique"],
        },
    },
    {
        "name": "call_evaluator",
        "description": "Execute the current model code and return accuracy metrics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "call_critic",
        "description": "Analyze current code and metrics. Returns improvement suggestions.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "finish_pipeline",
        "description": "Stop the optimization loop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why we are stopping",
                }
            },
            "required": ["reason"],
        },
    },
]


def run(state: MLPipelineState, cost_tracker: CostTracker) -> MLPipelineState:
    """Run the orchestration loop until done or max_iterations reached."""
    data_line = f"Data directory: {state.data_dir}\n" if state.data_dir else ""
    messages = [
        {
            "role": "user",
            "content": (
                f"Problem: {state.problem}\n"
                f"Target {state.metric}: {state.target_score}\n"
                f"Max iterations: {state.max_iterations}\n"
                f"{data_line}"
                "Begin the ML optimization pipeline."
            ),
        }
    ]

    iteration = 0
    current_approach = ""
    iter_input_tokens = 0
    iter_output_tokens = 0
    iter_cost = 0.0

    while True:
        status = _build_status(state, iteration)
        if status:
            messages.append({"role": "user", "content": status})

        print(f"\n[Orchestrator] Deciding next action (iter={iteration})...")
        response = get_client().messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        orch_in = response.usage.input_tokens
        orch_out = response.usage.output_tokens
        cost_tracker.record("Orchestrator", orch_in, orch_out)
        iter_input_tokens += orch_in
        iter_output_tokens += orch_out

        # Parse orchestrator decision
        tool_use = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use is None:
            print("[Orchestrator] No tool call — stopping.")
            break

        tool_name = tool_use.name
        tool_input = tool_use.input
        print(f"[Orchestrator] → {tool_name}")

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        if tool_name == "finish_pipeline":
            reason = tool_input.get("reason", "done")
            print(f"[Orchestrator] Finished: {reason}")
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "Done."}],
            })
            break

        elif tool_name == "call_builder":
            use_critique = tool_input.get("use_critique", False)
            critique_to_pass = state.current_critique if use_critique else ""
            code, approach = builder.run(state, critique_to_pass, cost_tracker)
            state.current_code = code
            current_approach = approach
            tool_result = f"Code generated. Approach: {approach}"

        elif tool_name == "call_evaluator":
            if not state.current_code:
                tool_result = "Error: no code to evaluate. Call call_builder first."
            else:
                metrics = evaluator.run(state.current_code, cost_tracker, state)
                state.current_metrics = metrics
                score = metrics.get(state.metric, 0.0)
                tool_result = json.dumps(metrics)

                # After evaluator, finalize this iteration
                iteration += 1
                iter_result = IterationResult(
                    iteration=iteration,
                    code=state.current_code,
                    approach=current_approach,
                    metrics=metrics,
                    critique=state.current_critique,
                    input_tokens=iter_input_tokens,
                    output_tokens=iter_output_tokens,
                    cost_usd=iter_cost,
                )
                state.history.append(iter_result)
                cost_tracker.print_iteration_summary(iteration, score)
                iter_input_tokens = 0
                iter_output_tokens = 0
                iter_cost = 0.0

                if score >= state.target_score:
                    print(f"[Orchestrator] Target {state.metric} {state.target_score} reached!")
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}],
                    })
                    break

                if iteration >= state.max_iterations:
                    print(f"[Orchestrator] Max iterations ({state.max_iterations}) reached.")
                    messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}],
                    })
                    break

        elif tool_name == "call_critic":
            if not state.current_code or not state.current_metrics:
                tool_result = "Error: need code and metrics first."
            else:
                history_summary = _build_history_summary(state)
                critique = critic.run(
                    state.current_code, state.current_metrics, history_summary, cost_tracker
                )
                state.current_critique = critique
                tool_result = critique

        else:
            tool_result = f"Unknown tool: {tool_name}"

        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result}],
        })

    return state


def _build_status(state: MLPipelineState, iteration: int) -> str:
    if not state.history:
        return ""
    m = state.metric
    latest = state.history[-1]
    best = max(r.metrics.get(m, 0) for r in state.history)
    return (
        f"[Status] Iteration {iteration} done. "
        f"Best {m}: {best:.4f}. "
        f"Latest: {latest.metrics.get(m, 0):.4f} ({latest.approach})"
    )


def _build_history_summary(state: MLPipelineState) -> str:
    if not state.history:
        return ""
    m = state.metric
    lines = []
    for r in state.history:
        score = r.metrics.get(m, 0.0)
        lines.append(f"- Iter {r.iteration}: {r.approach} → {m}={score:.4f}")
    return "\n".join(lines)
