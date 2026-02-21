import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

# --- IMAGE PREPROCESSING (CNN) ---
def get_image_transforms(is_train=True):
    """Returns torchvision transforms for production line images."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

class ProductionImageDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label

# --- SENSOR TIME-SERIES PREPROCESSING (LSTM) ---
def create_sliding_windows(data, targets, window_size):
    """
    Converts 2D tabular sensor data into 3D sequences for LSTM.
    data shape: (total_steps, num_sensors)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

class SensorSequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]