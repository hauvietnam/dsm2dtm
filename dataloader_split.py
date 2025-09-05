import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from utils import normalize_patch, calculate_patch_positions, is_valid_patch

class DSMSplitDataset(Dataset):
    """
    Dataset loader for split train/val/test folders
    Each split folder should contain images/ and masks/ subdirectories
    """
    def __init__(self, split_dir, patch_size=256, nodata_val=-9999, nodata_threshold=0.1, transforms=None):
        """
        Args:
            split_dir: Path to split directory (e.g., 'datasets_split/train')
            patch_size: Size of patches to extract
            nodata_val: Value representing no-data pixels
            nodata_threshold: Maximum ratio of nodata pixels allowed in patch
            transforms: Optional transforms to apply
        """
        self.images_dir = os.path.join(split_dir, "images")
        self.masks_dir = os.path.join(split_dir, "masks")
        self.patch_size = patch_size
        self.nodata_val = nodata_val
        self.nodata_threshold = nodata_threshold
        self.transforms = transforms

        # Verify directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Find matching image-mask pairs
        images_all = set(os.listdir(self.images_dir))
        masks_all = set(os.listdir(self.masks_dir))

        common_files = sorted(images_all.intersection(masks_all))
        if len(common_files) == 0:
            raise RuntimeError(f"No matching image-mask pairs found in {split_dir}!")

        self.image_files = common_files
        self.mask_files = common_files

        print(f"Found {len(common_files)} image-mask pairs in {split_dir}")

        # Prepare patch index
        self.patch_index = []
        self._prepare_patches()

    def _prepare_patches(self):
        print("Preparing patch indices...")
        for file_idx, filename in enumerate(self.image_files):
            image_path = os.path.join(self.images_dir, filename)
            with rasterio.open(image_path) as src:
                img = src.read(1)
            H, W = img.shape
            
            # Use shared utility for consistent patch positioning
            i_positions, j_positions = calculate_patch_positions(H, W, self.patch_size)

            for i_pos in i_positions:
                for j_pos in j_positions:
                    patch = img[i_pos:i_pos+self.patch_size, j_pos:j_pos+self.patch_size]
                    if is_valid_patch(patch, self.nodata_val, self.nodata_threshold):
                        self.patch_index.append((file_idx, i_pos, j_pos))
        
        print(f"Created {len(self.patch_index)} valid patches")

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        file_idx, top, left = self.patch_index[idx]

        img_path = os.path.join(self.images_dir, self.image_files[file_idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[file_idx])

        with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
            img = src_img.read(1)[top:top+self.patch_size, left:left+self.patch_size]
            mask = src_mask.read(1)[top:top+self.patch_size, left:left+self.patch_size]

        # Use shared normalization function
        img = normalize_patch(img, self.nodata_val)
        
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, mask_tensor

def get_split_dataloaders(dataset_root, batch_size=8, patch_size=256, nodata_val=-9999, 
                         nodata_threshold=0.1, num_workers=4, pin_memory=True):
    """
    Create dataloaders for train, validation, and test sets from split folders
    
    Args:
        dataset_root: Root directory containing train/, val/, test/ folders
        batch_size: Batch size for dataloaders
        patch_size: Size of patches
        nodata_val: No-data value
        nodata_threshold: Maximum no-data ratio in patches
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Paths to split directories
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")
    
    # Verify all directories exist
    for split_name, split_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{split_name} directory not found: {split_path}")
    
    print(f"Loading datasets from {dataset_root}")
    
    # Create datasets
    train_dataset = DSMSplitDataset(
        train_dir,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold
    )
    
    val_dataset = DSMSplitDataset(
        val_dir,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold
    )
    
    test_dataset = DSMSplitDataset(
        test_dir,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

def get_single_split_dataloader(dataset_root, split_name, batch_size=8, patch_size=256, 
                               nodata_val=-9999, nodata_threshold=0.1, num_workers=4, 
                               pin_memory=True, shuffle=None):
    """
    Create dataloader for a single split (train, val, or test)
    
    Args:
        dataset_root: Root directory containing split folders
        split_name: Name of split ('train', 'val', or 'test')
        batch_size: Batch size
        patch_size: Size of patches
        nodata_val: No-data value
        nodata_threshold: Maximum no-data ratio in patches
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data (None for auto-detection based on split)
    
    Returns:
        DataLoader for the specified split
    """
    
    split_dir = os.path.join(dataset_root, split_name)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"{split_name} directory not found: {split_dir}")
    
    # Auto-detect shuffle setting if not specified
    if shuffle is None:
        shuffle = (split_name == 'train')  # Only shuffle training data
    
    print(f"Loading {split_name} dataset from {split_dir}")
    
    dataset = DSMSplitDataset(
        split_dir,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split_name == 'train')  # Only drop last batch for training
    )

    return dataloader

# Example usage and testing
if __name__ == "__main__":
    # Test the split dataloader
    try:
        dataset_root = "datasets_split"  # Path to split datasets
        
        # Test single split loading
        train_loader = get_single_split_dataloader(dataset_root, 'train', batch_size=4)
        print(f"Train loader: {len(train_loader)} batches")
        
        # Test full split loading
        train_loader, val_loader, test_loader = get_split_dataloaders(
            dataset_root, 
            batch_size=4,
            num_workers=2
        )
        
        print(f"All loaders created successfully:")
        print(f"- Train: {len(train_loader)} batches")  
        print(f"- Val: {len(val_loader)} batches")
        print(f"- Test: {len(test_loader)} batches")
        
        # Test loading a batch
        for images, masks in train_loader:
            print(f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}")
            break
            
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        print("Make sure to run split_dataset.py first to create the split folders!")