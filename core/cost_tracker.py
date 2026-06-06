from core.client import INPUT_COST_PER_TOKEN, OUTPUT_COST_PER_TOKEN


class CostTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._iteration_calls: list[tuple[str, int, int]] = []

    def record(self, agent_name: str, input_tokens: int, output_tokens: int) -> float:
        cost = (
            input_tokens * INPUT_COST_PER_TOKEN
            + output_tokens * OUTPUT_COST_PER_TOKEN
        )
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self._iteration_calls.append((agent_name, input_tokens, output_tokens))
        print(
            f"  [{agent_name}] {input_tokens:,} in / {output_tokens:,} out"
            f"  (${cost:.4f})"
        )
        return cost

    def print_iteration_summary(self, iteration: int, accuracy: float | None = None) -> float:
        iter_in = sum(t for _, t, _ in self._iteration_calls)
        iter_out = sum(t for _, _, t in self._iteration_calls)
        iter_cost = (
            iter_in * INPUT_COST_PER_TOKEN + iter_out * OUTPUT_COST_PER_TOKEN
        )
        acc_str = f" | Accuracy: {accuracy:.4f}" if accuracy is not None else ""
        print(
            f"[Iter {iteration}] Total: {iter_in:,} in / {iter_out:,} out"
            f"  (${iter_cost:.4f}){acc_str}"
        )
        print("─" * 60)
        self._iteration_calls = []
        return iter_cost

    def print_total_summary(self):
        total_cost = (
            self.total_input_tokens * INPUT_COST_PER_TOKEN
            + self.total_output_tokens * OUTPUT_COST_PER_TOKEN
        )
        print("\n" + "═" * 60)
        print(f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out")
        print(f"Total cost:   ${total_cost:.4f}")
        print("═" * 60)
