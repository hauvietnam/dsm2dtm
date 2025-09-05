import torch
import numpy as np
import rasterio
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from safetensors.torch import load_file
from tqdm import tqdm
from utils import normalize_patch, patch_to_tensor, calculate_patch_positions, is_valid_patch

def create_patches(img, patch_size, nodata_val, nodata_threshold=0.1):
    """Create patches using shared utilities for consistency with training"""
    H, W = img.shape
    
    # Use shared utility for consistent patch positioning
    i_positions, j_positions = calculate_patch_positions(H, W, patch_size)

    patches = []
    for i_pos in i_positions:
        for j_pos in j_positions:
            patch = img[i_pos:i_pos+patch_size, j_pos:j_pos+patch_size]
            if is_valid_patch(patch, nodata_val, nodata_threshold):
                patches.append((i_pos, j_pos))
    print(f"Total patches after filtering nodata: {len(patches)}")
    return patches

def infer_dsm(dsm_path, safetensor_path, patch_size=256, nodata_val=-9999, nodata_threshold=0.1, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
        profile = src.profile

    patches_pos = create_patches(dsm, patch_size, nodata_val, nodata_threshold)

    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    config = UperNetConfig(backbone_config=backbone_config, num_labels=2)

    model = UperNetForSemanticSegmentation(config)
    state_dict = load_file(safetensor_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"Model loaded. Number of parameters: {sum(p.numel() for p in model.parameters())}")

    mask_pred = np.zeros_like(dsm, dtype=np.float32)
    count_mask = np.zeros_like(dsm, dtype=np.float32)

    with torch.no_grad():
        for idx, (top, left) in enumerate(tqdm(patches_pos, desc="Infer patches")):
            patch = dsm[top:top+patch_size, left:left+patch_size]
            
            # Use shared preprocessing for consistency
            normalized_patch = normalize_patch(patch, nodata_val)
            input_tensor = patch_to_tensor(normalized_patch).to(device)
            outputs = model(input_tensor)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

            # Simple weighted blending for overlapping regions
            h, w = pred.shape
            weights = np.ones((h, w), dtype=np.float32)
            
            # Reduce weight near edges to blend overlaps smoothly
            fade_size = 16  # pixels to fade from edges
            if fade_size > 0 and h > 2*fade_size and w > 2*fade_size:
                # Create fade zones at edges
                for i in range(fade_size):
                    alpha = (i + 1) / fade_size
                    weights[i, :] *= alpha          # top edge
                    weights[-(i+1), :] *= alpha     # bottom edge
                    weights[:, i] *= alpha          # left edge
                    weights[:, -(i+1)] *= alpha     # right edge

            # Accumulate predictions with weights
            mask_pred[top:top+patch_size, left:left+patch_size] += pred * weights
            count_mask[top:top+patch_size, left:left+patch_size] += weights

    # Average overlapping predictions and convert to binary mask
    count_mask[count_mask == 0] = 1
    mask_pred = mask_pred / count_mask
    mask_pred = (mask_pred >= 0.5).astype(np.uint8)

    out_path = dsm_path.replace(".TIF", "_pred_mask.tif")
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    if "nodata" in profile:
        del profile["nodata"]
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask_pred, 1)

    print(f"Saved prediction mask at: {out_path}")
    return out_path

if __name__ == "__main__":
    dsm_path = r"datasets\images\dsm_1.TIF"  # Updated relative path
    safetensor_path = r"pretrain\model.safetensors"  # Updated relative path
    infer_dsm(dsm_path, safetensor_path)
