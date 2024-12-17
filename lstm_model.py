import torch.nn as nn
from helpful_functions.useful_functions import get_device

# Get Device
device = get_device()

# LSTM Model with Logits Support
class LSTMD(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, output_activation=None, return_logits=True):
        """
        LSTM Model with optional activation and raw logits output.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of LSTM hidden units.
            num_stacked_layers (int): Number of stacked LSTM layers.
            output_activation (str, optional): Activation function ("sigmoid" or "relu"). Defaults to None.
            return_logits (bool): If True, return raw logits. Defaults to True.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.return_logits = return_logits

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
        """
        Forward pass for LSTM model.
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_size].
        Returns:
            Output logits or activated values (shape: [batch_size, 1]).
        """
        # LSTM forward pass
        out, _ = self.lstm(x)

        # Fully connected layer on the last time step
        out = self.fc(out[:, -1, :])  # Take only the last time step

        # Return raw logits if specified
        if self.return_logits:
            return out

        # Apply activation function (if any)
        if self.activation:
            out = self.activation(out)

        return out