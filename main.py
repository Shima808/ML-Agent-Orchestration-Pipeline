import argparse
import sys
from core.types import MLPipelineState
from core.cost_tracker import CostTracker
from agents import orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent ML optimization pipeline powered by Claude"
    )
    parser.add_argument(
        "--problem",
        default="Classify Iris flower species with maximum accuracy.",
        help="ML problem description",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of optimization iterations (default: 5)",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=0.97,
        help="Target accuracy for early stopping (default: 0.97)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ML Agent Orchestration Pipeline")
    print("=" * 60)
    print(f"  Problem:        {args.problem}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Target score:   {args.target_score}")
    print(f"  Model:          claude-sonnet-4-6")
    print("=" * 60 + "\n")

    state = MLPipelineState(
        problem=args.problem,
        max_iterations=args.max_iterations,
        target_score=args.target_score,
    )
    cost_tracker = CostTracker()

    try:
        state = orchestrator.run(state, cost_tracker)
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        raise

    # Final summary
    print("\n" + "=" * 60)
    print("  Final Results")
    print("=" * 60)
    if state.history:
        best = max(state.history, key=lambda r: r.metrics.get("accuracy", 0))
        print(f"  Best accuracy: {best.metrics.get('accuracy', 0):.4f}")
        print(f"  Best model:    {best.metrics.get('model', 'unknown')}")
        print(f"  At iteration:  {best.iteration}")
    else:
        print("  No iterations completed.")

    cost_tracker.print_total_summary()


if __name__ == "__main__":
    main()
