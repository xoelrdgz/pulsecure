from pathlib import Path
from typing import Optional

from ..training import export_to_json, export_to_rust, train_calibrated_model


def full_pipeline(
    output_dir: Optional[Path] = None,
    precision_bits: int = 12,
    min_recall: float = 0.95,
    verbose: bool = True,
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = train_calibrated_model(precision_bits=precision_bits, min_recall=min_recall)
    export_to_json(model, output_dir / "calibrated_model.json")
    export_to_rust(model, output_dir / "calibrated_model.rs")
    if verbose:
        print(f"Exported model to {output_dir}")
    return output_dir
