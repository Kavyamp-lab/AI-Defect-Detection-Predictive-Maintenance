import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DefectDetectorCNN(nn.Module):
    """
    ResNet-based CNN for binary Defect Detection (Defective vs. Normal).
    """
    def __init__(self, pretrained=True):
        super(DefectDetectorCNN, self).__init__()
        # Load pre-trained ResNet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        
        # Replace the final fully connected layer for binary classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1),
            nn.Sigmoid() # Output probability of defect
        )

    def forward(self, x):
        return self.model(x)


class PredictiveMaintenanceLSTM(nn.Module):
    """
    LSTM model for predicting Remaining Useful Life (RUL) or anomaly scores 
    based on sequential sensor data (temperature, vibration, pressure, etc.).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(PredictiveMaintenanceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for RUL regression prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Predicts single continuous value (e.g., hours left)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out