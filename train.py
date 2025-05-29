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
    dataset_root = "/kaggle/input/dsm2dtm/datasets"
    batch_size = 8
    patch_size = 256
    num_epochs = 200
    nodata_val = -9999
    nodata_threshold = 0.1
    number_of_classes = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        dataset_root,
        batch_size,
        patch_size,
        nodata_val,
        nodata_threshold,
        val_split=0.2,
        num_workers=4
    )

    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    config = UperNetConfig(backbone_config=backbone_config, num_labels=number_of_classes)
    model = UperNetForSemanticSegmentation(config)
    model.to(device)

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    best_miou = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loop(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_miou = validate_loop(val_loader, model, criterion, device, number_of_classes)

        if val_miou > best_miou:
            best_miou = val_miou
            save_path = "upernet_convnext_best_miou"
            model.save_pretrained(save_path)
            print(f"Model mới được lưu tại: {save_path} với mIoU: {val_miou:.4f}")
        else:
            print(f"mIoU không cải thiện ({val_miou:.4f} <= {best_miou:.4f}), không lưu model.")
