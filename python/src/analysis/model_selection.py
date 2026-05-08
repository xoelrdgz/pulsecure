from dataclasses import dataclass

from ..training.pipeline import train_calibrated_model


@dataclass
class ModelSelectionResult:
    name: str
    roc_auc: float
    average_precision: float
    brier: float


def evaluate_locked_v2() -> ModelSelectionResult:
    model = train_calibrated_model()
    return ModelSelectionResult(
        name="logistic_sigmoid_quantized",
        roc_auc=model.auc_quantized,
        average_precision=float("nan"),
        brier=model.brier_after,
    )


def main() -> None:
    result = evaluate_locked_v2()
    print(result)


if __name__ == "__main__":
    main()
