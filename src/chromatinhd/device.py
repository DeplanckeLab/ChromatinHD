import torch


def _get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


default_device = _get_default_device()


def get_default_device():
    global default_device
    return default_device


def set_default_device(device):
    global default_device
    default_device = device
