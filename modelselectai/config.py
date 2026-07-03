from dataclasses import dataclass

@dataclass
class RunConfig:
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_top_models: int = 5
    # suitability weight defaults (used when app not found)
    w_perf: float = 0.55
    w_train_time: float = 0.15
    w_infer_time: float = 0.15
    w_size: float = 0.05
    w_explain: float = 0.10
    # classification detection threshold for numeric target
    cls_cardinality_threshold: int = 20