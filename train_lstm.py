import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from helpful_functions.useful_functions import get_device
from lstm_model import LSTMD  # Import the LSTMD model class
from data_preprocessing import prepare_stock_data  # Import data preparation

# Device configuration
device = get_device()

# Prepare datasets
file_path = "data/tsla_monthly_sentiment_data.csv"
sequence_length = 10
print("Preparing datasets...")
train_dataset, test_dataset = prepare_stock_data(file_path, sequence_length=sequence_length)

# DataLoader for training and testing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the LSTM model, loss function, and optimizer
input_size = train_dataset[0][0].shape[1]  # Automatically get input size from dataset
hidden_size = 64
num_stacked_layers = 2

model = LSTMD(input_size, hidden_size, num_stacked_layers, output_activation="sigmoid").to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(model, train_loader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)  # Ensure targets are (batch_size, 1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

# Testing Loop
def test(model, test_loader, device):
    """
    Testing loop to calculate accuracy.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

# Train and Evaluate the Model
epochs = 20
print("\nStarting training...")
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    accuracy = test(model, test_loader, device)

    print(f"Epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")

print("\nTraining complete!")