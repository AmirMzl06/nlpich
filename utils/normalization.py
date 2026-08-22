import torch
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussSmooth(inputs, kernelSD=2, padding='SAME'):
    """
    Applies a 1D gaussian smoothing operation with PyTorch to smooth the data along the time axis.

    Args:
        inputs (numpy array : B x T x N or 1 x T x N): A 3d array with batch size B, time steps T, and number of features N
        kernelSD (float): standard deviation of the Gaussian smoothing kernel

    Returns:
        smoothedData (numpy array : B x T x N or 1 x T x N): A smoothed 3d array with batch size B, time steps T, and number of features N
    """

    inp = np.zeros([100], dtype=np.float32)
    inp[50] = 1
    gaussKernel = gaussian_filter1d(inp, kernelSD)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel/np.sum(gaussKernel))
    
    if inputs.ndim == 2:
        inputs = inputs[np.newaxis, :, :]
    
    B, T, C = inputs.shape
    
    kernel_tensor = torch.from_numpy(gaussKernel).float()
    kernel_tensor = kernel_tensor.view(1, 1, -1).repeat(C, 1, 1)
    
    inputs_tensor = torch.from_numpy(inputs).float()
    inputs_tensor = inputs_tensor.permute(0, 2, 1)
    
    if padding == 'SAME':
        padding_size = (kernel_tensor.shape[2] - 1) // 2
    else:
        padding_size = 0
        
    smoothed_tensor = F.conv1d(inputs_tensor, kernel_tensor, padding=padding_size, groups=C)
    smoothed_tensor = smoothed_tensor.permute(0, 2, 1)
    
    smoothed_data = smoothed_tensor.numpy()
    
    if inputs.shape[0] == 1:
        smoothed_data = smoothed_data[0]
    
    return smoothed_data

from scipy.spatial import cKDTree

def min_l2_distance(vectors: np.ndarray) -> float:
    """
    Minimum Euclidean distance between distinct rows, ignoring duplicates.
    Uses cKDTree for efficiency.
    """
    X = np.asarray(vectors, dtype=np.float32)
    X = np.unique(X, axis=0)           # drop duplicate rows
    if len(X) < 2:
        raise ValueError("Need at least 2 unique vectors.")

    tree = cKDTree(X)
    # k=2 -> first is the point itself (distance 0), second is nearest neighbor
    dists, _ = tree.query(X, k=2, workers=-1)
    return float(np.min(dists[:, 1]))