import numpy as np
from sklearn.metrics import precision_recall_curve


def probability_logit(prob):
    eps = 1e-6
    clipped = np.clip(prob, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def find_optimal_threshold(y_true, y_prob, min_recall=0.90):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    valid_idx = np.where(recalls[:-1] >= min_recall)[0]
    idx = valid_idx[-1] if len(valid_idx) > 0 else 0
    return thresholds[idx], recalls[idx], precisions[idx]


def main():
    from ..training.pipeline import train_calibrated_model

    model = train_calibrated_model()
    print(f"AUC calibrated: {model.auc_calibrated:.4f}")
    print(f"AUC quantized: {model.auc_quantized:.4f}")
    print(f"Recall: {model.recall_at_threshold:.4f}")


if __name__ == "__main__":
    main()
