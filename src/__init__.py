"""
Smart Manufacturing Core Package
Contains deep learning models, data preprocessing pipelines, and training routines.
"""

# Import key classes and functions from the module's files
from .models import DefectDetectorCNN, PredictiveMaintenanceLSTM
from .data_preprocessing import (
    ProductionImageDataset, 
    SensorSequenceDataset, 
    get_image_transforms, 
    create_sliding_windows
)
from .train import train_cnn, train_lstm

# Define what is available when someone uses `from src import *`
__all__ = [
    "DefectDetectorCNN",
    "PredictiveMaintenanceLSTM",
    "ProductionImageDataset",
    "SensorSequenceDataset",
    "get_image_transforms",
    "create_sliding_windows",
    "train_cnn",
    "train_lstm"
]