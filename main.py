import argparse
import sys
from core.types import MLPipelineState
from core.cost_tracker import CostTracker
from agents import orchestrator, planner


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent ML optimization pipeline powered by Claude"
    )
    parser.add_argument(
        "--problem",
        default=None,
        help="ML problem description (省略するとプランニングモードで起動)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of optimization iterations (default: 5)",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=None,
        help="Target score for early stopping (default: 0.97 for accuracy, 0.85 for auc)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric key to optimise: accuracy | auc | rmse … (default: accuracy)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing train.csv and test.csv for CSV-mode competitions",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Name of the target column in the CSV (required when --data-dir is set)",
    )
    parser.add_argument(
        "--submission-path",
        default=None,
        help="Path to write submission.csv (optional, used in competition mode)",
    )
    args = parser.parse_args()

    if args.problem is None:
        plan = planner.run()
        problem         = plan["problem"]
        max_iterations  = plan["max_iterations"]
        target_score    = plan["target_score"]
        metric          = plan["metric"]
        data_dir        = plan["data_dir"]
        target_col      = plan["target_col"]
        submission_path = plan["submission_path"]
    else:
        problem         = args.problem
        max_iterations  = args.max_iterations if args.max_iterations is not None else 5
        metric          = args.metric or "accuracy"
        target_score    = args.target_score if args.target_score is not None else (0.85 if metric == "auc" else 0.97)
        data_dir        = args.data_dir
        target_col      = args.target_col
        submission_path = args.submission_path

    print("=" * 60)
    print("  ML Agent Orchestration Pipeline")
    print("=" * 60)
    print(f"  Problem:        {problem}")
    print(f"  Metric:         {metric}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Target score:   {target_score}")
    if data_dir:
        print(f"  Data dir:       {data_dir}")
        print(f"  Target col:     {target_col}")
    if submission_path:
        print(f"  Submission:     {submission_path}")
    print(f"  Model:          claude-sonnet-4-6")
    print("=" * 60 + "\n")

    state = MLPipelineState(
        problem=problem,
        max_iterations=max_iterations,
        target_score=target_score,
        metric=metric,
        data_dir=data_dir,
        target_col=target_col,
        submission_path=submission_path,
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
        best = max(state.history, key=lambda r: r.metrics.get(state.metric, 0))
        print(f"  Best {state.metric}: {best.metrics.get(state.metric, 0):.4f}")
        print(f"  Best model:    {best.metrics.get('model', 'unknown')}")
        print(f"  At iteration:  {best.iteration}")
    else:
        print("  No iterations completed.")

    cost_tracker.print_total_summary()


if __name__ == "__main__":
    main()
