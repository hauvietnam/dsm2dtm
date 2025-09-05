import torch
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from torch.nn import CrossEntropyLoss
from dataloader_split import get_single_split_dataloader
from tqdm import tqdm
import os

def calculate_miou(preds, labels, num_classes, ignore_index=255):
    """Calculate mean IoU for batch."""
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = preds == cls
        label_inds = labels == cls

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    
    # Return mean excluding nan values
    ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)

def calculate_per_class_iou(preds, labels, num_classes, ignore_index=255):
    """Calculate IoU for each class separately."""
    class_ious = {}
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = preds == cls
        label_inds = labels == cls

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            class_ious[f'class_{cls}'] = float('nan')
        else:
            class_ious[f'class_{cls}'] = intersection / union
    
    return class_ious

def validate_loop(dataloader, model, criterion, device, num_classes, split_name="Validation"):
    """Comprehensive validation loop with detailed metrics."""
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    miou_sum = 0.0
    batches = 0
    
    # Per-class tracking
    class_iou_sums = {f'class_{i}': 0.0 for i in range(num_classes)}
    class_iou_counts = {f'class_{i}': 0 for i in range(num_classes)}

    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"{split_name}")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs.logits

            loss = criterion(logits, masks)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            # Overall mIoU
            batch_miou = calculate_miou(preds, masks, num_classes)
            miou_sum += batch_miou
            batches += 1
            
            # Per-class IoU
            batch_class_ious = calculate_per_class_iou(preds, masks, num_classes)
            for class_name, iou in batch_class_ious.items():
                if not torch.isnan(torch.tensor(iou)):
                    class_iou_sums[class_name] += iou
                    class_iou_counts[class_name] += 1

            loop.set_postfix(loss=loss.item(), mIoU=batch_miou)

    # Calculate final metrics
    avg_loss = val_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels
    mean_iou = miou_sum / batches if batches > 0 else 0.0
    
    # Calculate per-class mean IoU
    class_mean_ious = {}
    for class_name in class_iou_sums:
        if class_iou_counts[class_name] > 0:
            class_mean_ious[class_name] = class_iou_sums[class_name] / class_iou_counts[class_name]
        else:
            class_mean_ious[class_name] = 0.0
    
    # Print results
    print(f"\n📊 {split_name} Results:")
    print(f"{'='*50}")
    print(f"{split_name} Loss: {avg_loss:.4f}")
    print(f"Pixel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Overall mIoU: {mean_iou:.4f}")
    print(f"\nPer-Class IoU:")
    for class_name, iou in class_mean_ious.items():
        class_id = class_name.split('_')[1]
        class_label = "Ground" if class_id == "0" else "Non-Ground"  # Assuming binary classification
        print(f"  {class_label} (Class {class_id}): {iou:.4f}")
    
    return avg_loss, accuracy, mean_iou, class_mean_ious

def run_comprehensive_validation(dataset_root, model_path, splits=['val'], 
                                batch_size=8, patch_size=256, nodata_val=-9999, 
                                nodata_threshold=0.1, num_workers=2):
    """
    Run comprehensive validation on specified dataset splits.
    
    Args:
        dataset_root: Root directory containing train/val/test folders
        model_path: Path to trained model directory
        splits: List of splits to evaluate ['val', 'test', 'train']
        batch_size: Batch size for evaluation
        patch_size: Patch size
        nodata_val: No-data value
        nodata_threshold: No-data threshold
        num_workers: Number of data loading workers
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    try:
        backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
        config = UperNetConfig(backbone_config=backbone_config, num_labels=2)
        model = UperNetForSemanticSegmentation.from_pretrained(model_path)
        model.to(device)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    criterion = CrossEntropyLoss(ignore_index=255)
    
    results = {}
    
    # Evaluate each split
    for split in splits:
        print(f"\n🔍 Evaluating {split} split...")
        
        try:
            # Load split dataloader
            dataloader = get_single_split_dataloader(
                dataset_root=dataset_root,
                split_name=split,
                batch_size=batch_size,
                patch_size=patch_size,
                nodata_val=nodata_val,
                nodata_threshold=nodata_threshold,
                num_workers=num_workers,
                shuffle=False  # No shuffling for evaluation
            )
            
            print(f"Dataset size: {len(dataloader.dataset)} patches, {len(dataloader)} batches")
            
            # Run validation
            loss, accuracy, miou, class_ious = validate_loop(
                dataloader, model, criterion, device, 2, split_name=split.upper()
            )
            
            results[split] = {
                'loss': loss,
                'accuracy': accuracy, 
                'miou': miou,
                'class_ious': class_ious,
                'num_samples': len(dataloader.dataset)
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {split} split: {e}")
            continue
    
    # Print summary
    if results:
        print(f"\n🎯 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Split':<10} {'Loss':<8} {'Accuracy':<10} {'mIoU':<8} {'Samples':<8}")
        print(f"{'-'*60}")
        
        for split_name, metrics in results.items():
            print(f"{split_name.upper():<10} {metrics['loss']:<8.4f} "
                  f"{metrics['accuracy']:<10.4f} {metrics['miou']:<8.4f} "
                  f"{metrics['num_samples']:<8}")
        
        # Save results to file
        results_file = "evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write("Evaluation Results\n")
            f.write("==================\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {dataset_root}\n\n")
            
            for split_name, metrics in results.items():
                f.write(f"{split_name.upper()} Split Results:\n")
                f.write(f"  Loss: {metrics['loss']:.4f}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n") 
                f.write(f"  mIoU: {metrics['miou']:.4f}\n")
                f.write(f"  Samples: {metrics['num_samples']}\n")
                f.write(f"  Per-class IoU:\n")
                for class_name, iou in metrics['class_ious'].items():
                    f.write(f"    {class_name}: {iou:.4f}\n")
                f.write("\n")
        
        print(f"\n📄 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    # Configuration
    dataset_root = "datasets_split"  # Root folder containing train/val/test
    model_path = "pretrain"  # Path to trained model
    
    # Evaluation settings
    batch_size = 8
    patch_size = 256
    nodata_val = -9999
    nodata_threshold = 0.1
    num_workers = 2
    
    # Check if dataset and model exist
    if not os.path.exists(dataset_root):
        print(f"❌ Dataset not found at: {dataset_root}")
        print("Please run 'python split_dataset.py' first to create the split dataset!")
        exit(1)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train a model first using 'python train_split.py'!")
        exit(1)
    
    print("🎯 DSM to DTM Model Evaluation")
    print("=" * 50)
    print(f"Dataset: {dataset_root}")
    print(f"Model: {model_path}")
    print(f"Batch size: {batch_size}")
    
    # Ask user which splits to evaluate
    print(f"\nAvailable splits to evaluate:")
    available_splits = []
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            available_splits.append(split)
            print(f"  ✅ {split}")
        else:
            print(f"  ❌ {split} (not found)")
    
    if not available_splits:
        print("❌ No valid splits found!")
        exit(1)
    
    # Default to validation and test sets
    splits_to_evaluate = ['val', 'test'] if all(s in available_splits for s in ['val', 'test']) else available_splits
    
    print(f"\n🚀 Evaluating splits: {splits_to_evaluate}")
    
    # Run comprehensive validation
    results = run_comprehensive_validation(
        dataset_root=dataset_root,
        model_path=model_path,
        splits=splits_to_evaluate,
        batch_size=batch_size,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold,
        num_workers=num_workers
    )
    
    print(f"\n✅ Evaluation completed!")