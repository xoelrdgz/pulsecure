"""
Pulsecure: Privacy-Preserving CVD Risk Prediction.

FHE-compatible model training and export for tfhe-rs.
"""

from .data import FEATURE_NAMES, NHANESData, get_preprocessed_data
from .training import CalibratedQuantizedModel, train_calibrated_model
from .export import full_pipeline

__all__ = [
    "FEATURE_NAMES",
    "NHANESData",
    "CalibratedQuantizedModel",
    "get_preprocessed_data",
    "train_calibrated_model",
    "full_pipeline",
]

__version__ = "1.0.0"
