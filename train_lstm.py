import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from lstm_model import LSTMD
from helpful_functions.useful_functions import get_device

# =====================
# DEVICE CONFIGURATION
# =====================
device = get_device()
print(f"Using device: {device}")

# =====================
# DATA LOADING
# =====================
class StockDataSet(Dataset):
    def __init__(self, features_path, target_path, sequence_length=10):
        features = pd.read_csv(features_path).values
        targets = pd.read_csv(target_path).values.flatten()
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Paths
features_path = "data/scaled_features.csv"
target_path = "data/target.csv"
sequence_length = 10

# Load dataset
dataset = StockDataSet(features_path, target_path, sequence_length)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Class balancing
targets = pd.read_csv(target_path).values.flatten()
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=targets)
pos_weight = torch.tensor([class_weights[1]], dtype=torch.float32).to(device)

# =====================
# BIDIRECTIONAL LSTM WITH DROPOUT
# =====================
class LSTMImproved(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # Extra dense layer
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

# =====================
# TRAINING AND EVALUATION
# =====================
def train_epoch(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    epoch_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            val_loss += criterion(output, y).item()
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    return val_loss / len(loader), correct / total

# =====================
# FINAL TRAINING LOOP
# =====================
model = LSTMImproved(input_size=dataset[0][0].shape[1], hidden_size=128, num_layers=2, dropout=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_val_acc = 0
for epoch in range(1, 30 + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_with_reg.pth")
        print(f"New best model saved with Val Accuracy: {val_acc:.4f}")

# Test evaluation
model.load_state_dict(torch.load("best_model_with_reg.pth"))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
