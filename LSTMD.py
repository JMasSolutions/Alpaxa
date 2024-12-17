import torch.nn as nn
from helpful_functions.useful_functions import get_device

# Get Device
device = get_device()

# LSTM Model
class LSTMD(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, output_activation=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

        # Optional activation function for output
        if output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif output_activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        # LSTM forward pass (hidden states default to zero)
        out, _ = self.lstm(x)

        # Fully connected layer on the last time step
        out = self.fc(out[:, -1, :])

        # Apply activation function (if any)
        if self.activation:
            out = self.activation(out)

        return out


# # Example of model usage
# input_size = 10      # Number of features (matches your input data)
# hidden_size = 64     # Number of LSTM hidden units
# num_stacked_layers = 2
#
# # Instantiate the model
# model = LSTMD(input_size, hidden_size, num_stacked_layers, output_activation="sigmoid")
# model = model.to(device)
