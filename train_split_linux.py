import os
import torch
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataloader_split import get_split_dataloaders

def calculate_miou(preds, labels, num_classes, ignore_index=255):
    """Tính mean IoU cho batch."""
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
            iou = float('nan')  # lớp không có trong batch
        else:
            iou = intersection / union
        ious.append(iou)
    # Trả về trung bình bỏ qua nan
    ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)

def train_loop(dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training")

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss

def validate_loop(dataloader, model, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    miou_sum = 0.0
    batches = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation")
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

            batch_miou = calculate_miou(preds, masks, num_classes)
            miou_sum += batch_miou
            batches += 1

            loop.set_postfix(loss=loss.item(), mIoU=batch_miou)

    avg_loss = val_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels
    mean_iou = miou_sum / batches if batches > 0 else 0.0
    print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {accuracy:.4f}, mIoU: {mean_iou:.4f}")
    return avg_loss, accuracy, mean_iou

def test_loop(dataloader, model, criterion, device, num_classes):
    """Test loop for final evaluation"""
    print("\n🧪 Running test evaluation...")
    model.eval()
    test_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    miou_sum = 0.0
    batches = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Testing")
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs.logits

            loss = criterion(logits, masks)
            test_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            batch_miou = calculate_miou(preds, masks, num_classes)
            miou_sum += batch_miou
            batches += 1

            loop.set_postfix(loss=loss.item(), mIoU=batch_miou)

    avg_loss = test_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels
    mean_iou = miou_sum / batches if batches > 0 else 0.0
    
    print(f"\n📊 Test Results:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Pixel Accuracy: {accuracy:.4f}")
    print(f"Test mIoU: {mean_iou:.4f}")
    
    return avg_loss, accuracy, mean_iou

if __name__ == "__main__":
    # Environment-specific configurations
    if os.path.exists("/kaggle"):
        # Kaggle environment
        dataset_root = "/kaggle/input/dsm2dtm-split/datasets_split"
        output_dir = "/kaggle/working"
        num_workers = 4
        batch_size = 16  # Larger batch for Kaggle GPUs
        print("🔥 Running on Kaggle environment")
    else:
        # Linux environment
        dataset_root = "datasets_split"
        output_dir = "output"
        num_workers = 8
        batch_size = 12  # Medium batch for Linux
        print("🐧 Running on Linux environment")
    
    # Training hyperparameters - Production settings
    patch_size = 256
    num_epochs = 200  # Full training
    nodata_val = -9999
    nodata_threshold = 0.1
    number_of_classes = 2
    
    # Learning rate schedule
    initial_lr = 5e-5
    lr_decay_step = 50
    lr_decay_factor = 0.5
    
    # Early stopping
    patience = 30
    min_delta = 0.001  # Minimum improvement threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if split dataset exists
    if not os.path.exists(dataset_root):
        print(f"❌ Split dataset not found at {dataset_root}")
        if os.path.exists("/kaggle"):
            print("Please add the split dataset to your Kaggle notebook!")
        else:
            print("Please run 'python split_dataset.py' first to create the split dataset!")
        exit(1)

    # Load split datasets
    print("Loading split datasets...")
    try:
        train_loader, val_loader, test_loader = get_split_dataloaders(
            dataset_root,
            batch_size=batch_size,
            patch_size=patch_size,
            nodata_val=nodata_val,
            nodata_threshold=nodata_threshold,
            num_workers=num_workers
        )
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        exit(1)

    print(f"\n📈 Dataset Statistics:")
    print(f"Training patches: {len(train_loader.dataset)}")
    print(f"Validation patches: {len(val_loader.dataset)}")
    print(f"Test patches: {len(test_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    config = UperNetConfig(backbone_config=backbone_config, num_labels=number_of_classes)
    model = UperNetForSemanticSegmentation(config)
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

    # Training tracking
    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_mious = []

    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Learning rate schedule: {initial_lr} → decay by {lr_decay_factor} every {lr_decay_step} epochs")

    for epoch in range(num_epochs):
        print(f"\n" + "="*80)
        print(f"Epoch {epoch+1}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("="*80)
        
        # Training phase
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss, val_acc, val_miou = validate_loop(val_loader, model, criterion, device, number_of_classes)
        val_losses.append(val_loss)
        val_mious.append(val_miou)
        
        # Update learning rate
        scheduler.step()

        # Save best model
        if val_miou > best_miou + min_delta:
            best_miou = val_miou
            best_epoch = epoch + 1
            patience_counter = 0
            
            save_path = os.path.join(output_dir, "best_model")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)
            
            print(f"✅ New best model saved! mIoU: {val_miou:.4f} (Epoch {best_epoch})")
            
            # Save PyTorch checkpoint as well
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }, os.path.join(save_path, 'checkpoint.pth'))
            
        else:
            patience_counter += 1
            print(f"mIoU: {val_miou:.4f} (Best: {best_miou:.4f} at epoch {best_epoch}) - Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
            break

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            model.save_pretrained(checkpoint_path)
            print(f"💾 Checkpoint saved at epoch {epoch+1}")

    print(f"\n🎉 Training completed!")
    print(f"Best validation mIoU: {best_miou:.4f} achieved at epoch {best_epoch}")

    # Load best model for testing
    print(f"\n📋 Loading best model for final test...")
    best_model_path = os.path.join(output_dir, "best_model")
    model = UperNetForSemanticSegmentation.from_pretrained(best_model_path)
    model.to(device)

    # Final test evaluation
    test_loss, test_acc, test_miou = test_loop(test_loader, model, criterion, device, number_of_classes)

    # Save comprehensive training summary
    summary_path = os.path.join(output_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Training Summary - Split Dataset\n")
        f.write("=================================\n")
        f.write(f"Dataset: {dataset_root}\n")
        f.write(f"Total Epochs: {len(train_losses)}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Early Stopping Patience: {patience}\n")
        f.write(f"Final Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n")
        f.write(f"\nDataset Statistics:\n")
        f.write(f"- Training patches: {len(train_loader.dataset)}\n")
        f.write(f"- Validation patches: {len(val_loader.dataset)}\n")
        f.write(f"- Test patches: {len(test_loader.dataset)}\n")
        f.write(f"\nBest Validation Results:\n")
        f.write(f"- Best Validation mIoU: {best_miou:.4f}\n")
        f.write(f"\nFinal Test Results:\n")
        f.write(f"- Test Loss: {test_loss:.4f}\n")
        f.write(f"- Test Accuracy: {test_acc:.4f}\n")
        f.write(f"- Test mIoU: {test_miou:.4f}\n")
        f.write(f"\nTraining History (Epoch, Train Loss, Val Loss, Val mIoU):\n")
        for i, (tl, vl, vm) in enumerate(zip(train_losses, val_losses, val_mious)):
            f.write(f"{i+1},{tl:.4f},{vl:.4f},{vm:.4f}\n")

    # Save training history as CSV
    csv_path = os.path.join(output_dir, "training_history.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_miou\n")
        for i, (tl, vl, vm) in enumerate(zip(train_losses, val_losses, val_mious)):
            f.write(f"{i+1},{tl:.4f},{vl:.4f},{vm:.4f}\n")

    print(f"\n📄 Results saved:")
    print(f"   Training summary: {summary_path}")
    print(f"   Training history: {csv_path}")
    print(f"   Best model: {best_model_path}")
    print(f"\n🎯 Final Performance:")
    print(f"   Best Validation mIoU: {best_miou:.4f}")
    print(f"   Final Test mIoU: {test_miou:.4f}")
    print(f"   Improvement over validation: {test_miou - best_miou:+.4f}")
    
    if test_miou > best_miou:
        print("✅ Model generalizes well! Test performance exceeds validation.")
    else:
        print("⚠️  Model may be slightly overfit. Consider regularization.")