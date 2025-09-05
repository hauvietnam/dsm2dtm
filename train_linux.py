import os
import torch
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataloader import DSMPatchFolderDataset
from torch.utils.data import DataLoader, random_split

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

def get_train_val_loaders(root_dir, batch_size, patch_size, nodata_val, nodata_threshold, val_split=0.2, num_workers=4):
    dataset = DSMPatchFolderDataset(
        root_dir,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold
    )

    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":
    # Kaggle/Linux specific configurations
    dataset_root = "/kaggle/input/dsm2dtm/datasets"  # Kaggle dataset path
    output_dir = "/kaggle/working"  # Kaggle output directory
    
    # Training hyperparameters - optimized for Kaggle environment
    batch_size = 16  # Larger batch size for better GPU utilization
    patch_size = 256
    num_epochs = 200  # Full training epochs for production model
    nodata_val = -9999
    nodata_threshold = 0.1
    number_of_classes = 2
    
    # Learning rate schedule
    initial_lr = 5e-5
    lr_decay_step = 50  # Decay every 50 epochs
    lr_decay_factor = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if running on Kaggle
    if os.path.exists("/kaggle"):
        print("Running on Kaggle environment")
        num_workers = 4  # Kaggle supports more workers
    else:
        print("Running on Linux environment")
        num_workers = 8  # Linux can handle more workers
        # Fallback paths for non-Kaggle Linux
        if not os.path.exists(dataset_root):
            dataset_root = "datasets"
            output_dir = "output"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loader, val_loader = get_train_val_loaders(
        dataset_root,
        batch_size,
        patch_size,
        nodata_val,
        nodata_threshold,
        val_split=0.2,
        num_workers=num_workers
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    config = UperNetConfig(backbone_config=backbone_config, num_labels=number_of_classes)
    model = UperNetForSemanticSegmentation(config)
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_factor)

    best_miou = 0.0
    best_epoch = 0
    patience = 20  # Early stopping patience
    patience_counter = 0

    # Training history for logging
    train_losses = []
    val_losses = []
    val_mious = []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        train_loop(train_loader, model, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, val_miou = validate_loop(val_loader, model, criterion, device, number_of_classes)
        
        # Update scheduler
        scheduler.step()
        
        # Save training history
        val_losses.append(val_loss)
        val_mious.append(val_miou)

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save model
            save_path = os.path.join(output_dir, "best_model")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_path)
            
            print(f"✅ New best model saved! mIoU: {val_miou:.4f} (Epoch {best_epoch})")
            
            # Also save as safetensors for inference
            safetensor_path = os.path.join(save_path, "model.safetensors")
            torch.save(model.state_dict(), safetensor_path.replace('.safetensors', '_pytorch.pth'))
            
        else:
            patience_counter += 1
            print(f"mIoU: {val_miou:.4f} (Best: {best_miou:.4f} at epoch {best_epoch})")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
            break
            
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            model.save_pretrained(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print(f"\n🎉 Training completed!")
    print(f"Best mIoU: {best_miou:.4f} achieved at epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(output_dir, 'best_model')}")
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.txt")
    with open(history_path, "w") as f:
        f.write("Epoch,Val_Loss,Val_mIoU\n")
        for i, (loss, miou) in enumerate(zip(val_losses, val_mious)):
            f.write(f"{i+1},{loss:.4f},{miou:.4f}\n")
    print(f"Training history saved to: {history_path}")