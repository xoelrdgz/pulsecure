"""Model export module for FHE inference with tfhe-rs."""

from pathlib import Path
from typing import Optional

from ..training import CalibratedQuantizedModel, train_calibrated_model, export_to_rust, export_to_json

def full_pipeline(
    output_dir: Optional[Path] = None,
    precision_bits: int = 12,
    min_recall: float = 0.90,
    verbose: bool = True,
) -> Path:
    """
    Train, calibrate, quantize and export model for Rust/FHE.
    
    Returns:
        Path to exported model directory.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent.parent / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("PULSECURE - FHE MODEL PIPELINE")
        print("=" * 60)
    
    # Train and quantize
    model = train_calibrated_model(precision_bits=precision_bits, min_recall=min_recall)
    
    # Export
    json_path = output_dir / "model.json"
    rust_path = output_dir / "model.rs"
    
    export_to_json(model, json_path)
    export_to_rust(model, rust_path)
    
    if verbose:
        print(f"\nExported to: {output_dir}")
        print(f"  - {json_path.name}")
        print(f"  - {rust_path.name}")
    
    return output_dir

if __name__ == "__main__":
    full_pipeline(verbose=True)
