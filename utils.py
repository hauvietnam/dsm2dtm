import numpy as np
import torch

def normalize_patch(patch, nodata_val=-9999):
    """Normalize patch data consistently across train and inference"""
    # Replace nodata values with 0
    patch = np.where(patch == nodata_val, 0, patch)
    
    # Normalize to [0, 1] range
    patch_min = patch.min()
    patch_max = patch.max()
    
    # Avoid division by zero
    if patch_max > patch_min:
        patch = (patch - patch_min) / (patch_max - patch_min)
    else:
        patch = np.zeros_like(patch)
    
    return patch

def patch_to_tensor(patch):
    """Convert normalized patch to tensor format for model input"""
    # Convert to tensor and repeat channels (grayscale to RGB)
    tensor = torch.tensor(patch, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
    return tensor.unsqueeze(0)  # (1, 3, H, W)

def calculate_patch_positions(H, W, patch_size, stride=None):
    """Calculate patch positions with consistent overlap strategy"""
    if stride is None:
        stride = patch_size // 2  # 50% overlap by default
    
    ps = patch_size
    
    # Calculate i positions (rows)
    i_positions = []
    i = 0
    while i + ps <= H:
        i_positions.append(i)
        i += stride
    
    # Add final position if needed to cover the edge
    if len(i_positions) == 0 or i_positions[-1] + ps < H:
        if H >= ps:
            i_positions.append(H - ps)
    
    # Calculate j positions (columns)
    j_positions = []
    j = 0
    while j + ps <= W:
        j_positions.append(j)
        j += stride
    
    # Add final position if needed to cover the edge
    if len(j_positions) == 0 or j_positions[-1] + ps < W:
        if W >= ps:
            j_positions.append(W - ps)
    
    return i_positions, j_positions

def is_valid_patch(patch, nodata_val, nodata_threshold):
    """Check if patch has enough valid data"""
    nodata_ratio = np.sum(patch == nodata_val) / patch.size
    return nodata_ratio < nodata_threshold