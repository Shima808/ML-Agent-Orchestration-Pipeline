from dataclasses import dataclass, field


@dataclass
class IterationResult:
    iteration: int
    code: str
    approach: str
    metrics: dict
    critique: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class MLPipelineState:
    problem: str
    max_iterations: int
    target_score: float
    metric: str = "accuracy"            # key in output JSON ("accuracy", "auc", "rmse", …)
    data_dir: str | None = None         # path to directory with train.csv / test.csv
    target_col: str | None = None       # target column name in the CSV
    submission_path: str | None = None  # where to write submission.csv
    history: list[IterationResult] = field(default_factory=list)
    current_code: str = ""
    current_metrics: dict = field(default_factory=dict)
    current_critique: str = ""
