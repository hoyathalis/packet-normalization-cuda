"""
PyTorch baseline implementation of per-packet normalization.
"""
import torch


def normalize_pytorch(x):
    """
    Normalize each row (packet) independently.
    
    Formula: x_norm = (x - mean) / (std + epsilon)
    
    Args:
        x: torch.Tensor of shape [num_packets, packet_len]
    
    Returns:
        Normalized tensor of same shape
    """
    # Compute mean and std along packet dimension (dim=1)
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False)  # Use population std for consistency
    
    # Normalize with small epsilon to prevent division by zero
    epsilon = 1e-5
    return (x - mean) / (std + epsilon)

