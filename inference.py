import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from helpful_functions.useful_functions import get_device

# =====================
# DEVICE CONFIGURATION
# =====================
device = get_device()
print(f"Using device: {device}")

# =====================
# DEFINE LSTM MODEL
# =====================
class LSTMD(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):  # Set num_layers to 3
        """
        LSTM Model for binary classification.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# =====================
# LOAD THE MODEL
# =====================
input_size = 10  # Adjust to match your feature dimension
hidden_size = 128
num_layers = 3  # Updated to match the saved model architecture

# Instantiate model and load checkpoint
model = LSTMD(input_size, hidden_size, num_layers, dropout=0.3).to(device)
model.load_state_dict(torch.load("models/76mod.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# =====================
# DATA PREPROCESSING FOR INFERENCE
# =====================
def preprocess_input(input_path, sequence_length=10):
    """
    Preprocess the input data for inference.
    Args:
        input_path (str): Path to the preprocessed feature file.
        sequence_length (int): Length of the sequence for LSTM.
    Returns:
        torch.Tensor: Processed input tensor ready for the model.
    """
    data = pd.read_csv(input_path).values  # Load CSV as numpy array
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i + sequence_length])
    X = torch.tensor(np.array(X), dtype=torch.float32).to(device)
    return X[-1].unsqueeze(0)  # Use the last sequence for inference

# Path to input data
input_data_path = "data/scaled_features.csv"
sequence_length = 10
input_tensor = preprocess_input(input_data_path, sequence_length)
print("Input tensor shape:", input_tensor.shape)

# =====================
# INFERENCE
# =====================
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.sigmoid(output).item()  # Sigmoid activation for binary classification

# Threshold for binary classification
threshold = 0.5
predicted_class = 1 if prediction > threshold else 0

# Print results
print(f"Predicted Probability: {prediction:.4f}")
print(f"Predicted Class: {predicted_class}")