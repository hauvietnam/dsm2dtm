import os
import shutil
import random
from pathlib import Path
import argparse

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test folders
    
    Args:
        source_dir: Path to original datasets folder with images/ and masks/ subdirs
        output_dir: Path where to create train/val/test folders
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15) 
        test_ratio: Proportion for test set (default: 0.15)
        seed: Random seed for reproducible splits
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # Source paths
    source_images_dir = os.path.join(source_dir, "images")
    source_masks_dir = os.path.join(source_dir, "masks")
    
    if not os.path.exists(source_images_dir):
        raise FileNotFoundError(f"Images directory not found: {source_images_dir}")
    if not os.path.exists(source_masks_dir):
        raise FileNotFoundError(f"Masks directory not found: {source_masks_dir}")
    
    # Get all image files
    image_files = set(os.listdir(source_images_dir))
    mask_files = set(os.listdir(source_masks_dir))
    
    # Find common files (images that have corresponding masks)
    common_files = sorted(list(image_files.intersection(mask_files)))
    
    if len(common_files) == 0:
        raise ValueError("No matching image-mask pairs found!")
    
    print(f"Found {len(common_files)} matching image-mask pairs")
    
    # Shuffle files for random split
    random.shuffle(common_files)
    
    # Calculate split indices
    total_files = len(common_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split files
    train_files = common_files[:train_end]
    val_files = common_files[train_end:val_end]
    test_files = common_files[val_end:]
    
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create output directories
    splits = {
        'train': train_files,
        'val': val_files, 
        'test': test_files
    }
    
    for split_name, file_list in splits.items():
        # Create directories
        split_images_dir = os.path.join(output_dir, split_name, "images")
        split_masks_dir = os.path.join(output_dir, split_name, "masks")
        
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_masks_dir, exist_ok=True)
        
        print(f"\nCopying {len(file_list)} files to {split_name} set...")
        
        # Copy files
        for filename in file_list:
            # Copy image
            src_img = os.path.join(source_images_dir, filename)
            dst_img = os.path.join(split_images_dir, filename)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            
            # Copy mask  
            src_mask = os.path.join(source_masks_dir, filename)
            dst_mask = os.path.join(split_masks_dir, filename)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
        
        print(f"✅ {split_name} set created: {len(file_list)} files")
    
    # Create a split info file
    info_file = os.path.join(output_dir, "split_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Dataset Split Information\n")
        f.write(f"========================\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Train: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%)\n")
        f.write(f"Validation: {len(val_files)} files ({len(val_files)/total_files*100:.1f}%)\n") 
        f.write(f"Test: {len(test_files)} files ({len(test_files)/total_files*100:.1f}%)\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"\nSplit ratios:\n")
        f.write(f"- Train: {train_ratio}\n")
        f.write(f"- Validation: {val_ratio}\n")
        f.write(f"- Test: {test_ratio}\n")
    
    print(f"\n🎉 Dataset split completed successfully!")
    print(f"Split info saved to: {info_file}")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Split DSM dataset into train/val/test folders')
    parser.add_argument('--source', '-s', default='datasets', 
                       help='Source dataset directory (default: datasets)')
    parser.add_argument('--output', '-o', default='datasets_split',
                       help='Output directory for split datasets (default: datasets_split)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15, 
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually copying files')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be copied")
        print(f"Source: {args.source}")
        print(f"Output: {args.output}")
        print(f"Ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
        
        # Check if source exists and count files
        source_images_dir = os.path.join(args.source, "images")
        source_masks_dir = os.path.join(args.source, "masks")
        
        if os.path.exists(source_images_dir) and os.path.exists(source_masks_dir):
            image_files = set(os.listdir(source_images_dir))
            mask_files = set(os.listdir(source_masks_dir))
            common_files = image_files.intersection(mask_files)
            print(f"Found {len(common_files)} matching image-mask pairs")
            
            total = len(common_files)
            train_count = int(total * args.train_ratio)
            val_count = int(total * args.val_ratio)
            test_count = total - train_count - val_count
            
            print(f"Would create:")
            print(f"- Train: {train_count} files")
            print(f"- Val: {val_count} files") 
            print(f"- Test: {test_count} files")
        else:
            print("❌ Source directories not found!")
        return
    
    try:
        split_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())