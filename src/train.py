import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# ==============================
# CNN TRAINING (UNCHANGED)
# ==============================
def train_cnn(model, train_loader, val_loader, epochs=10, lr=1e-4, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss, correct = 0.0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                val_loss += criterion(outputs, labels).item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {correct/len(val_loader.dataset):.4f}")

    torch.save(model.state_dict(), 'cnn_defect_model.pth')
    print("CNN Model saved successfully.")


# ==============================
# LSTM TRAINING
# ==============================
def train_lstm(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)

                outputs = model(sequences).squeeze()
                val_loss += criterion(outputs, targets).item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train MSE: {train_loss/len(train_loader):.4f} | "
              f"Val MSE: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), 'lstm_rul_model.pth')
    print("LSTM Model saved successfully.")


# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    import pandas as pd
    from models import PredictiveMaintenanceLSTM
    from data_preprocessing import create_sliding_windows, SensorSequenceDataset

    print("Loading C-MAPSS dataset...")

    data_path = "C:/Users/Kavya Patagar/OneDrive/Desktop/smart_manufacturing/data/predictive/train_FD001.txt"

    # ✅ Correct loading for C-MAPSS (space separated)
    df = pd.read_csv(data_path, sep="\s+", header=None)

    # Drop empty columns if any
    df = df.dropna(axis=1)

    # Convert all to float
    df = df.astype(float)

    # Columns:
    # 0 = engine_id
    # 1 = cycle
    # 2+ = sensors

    # Generate RUL
    df["RUL"] = df.groupby(0)[1].transform(lambda x: x.max() - x)

    # Extract sensor data only
    sensor_data = df.iloc[:, 2:-1].values
    targets = df["RUL"].values

    print("Creating sliding windows...")

    window_size = 10
    X, y = create_sliding_windows(sensor_data, targets, window_size)

    split = int(0.8 * len(X))

    train_dataset = SensorSequenceDataset(X[:split], y[:split])
    val_dataset = SensorSequenceDataset(X[split:], y[split:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    print("Creating LSTM model...")

    input_size = X.shape[2]
    model = PredictiveMaintenanceLSTM(input_size=input_size)

    print("Starting training...")

    train_lstm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=1e-3,
        device="cpu"
    )

    print("Training completed successfully!")