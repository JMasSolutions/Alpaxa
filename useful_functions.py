import torch

def get_device():
    if torch.mps.is_available():
        device = torch.device("mps")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    else:
        device = torch.device("cpu")
        return device