import torch
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from torch.nn import CrossEntropyLoss
from dataloader import get_dataloader  # import hàm get_dataloader từ dataloader.py
from tqdm import tqdm

def validate_loop(dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

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
            total_pixels += torch.numel(masks)

            loop.set_postfix(loss=loss.item())

    avg_loss = val_loss / len(dataloader)
    accuracy = correct_pixels / total_pixels
    print(f"Validation Loss: {avg_loss:.4f}, Pixel Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

if __name__ == "__main__":
    dataset_root = "path_to_val_dataset"  # đường dẫn folder validation
    batch_size = 8
    patch_size = 256
    nodata_val = -9999
    nodata_threshold = 0.1
    number_of_classes = 5  # chỉnh lại số lớp của bạn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataloader = get_dataloader(
        dataset_root,
        batch_size=batch_size,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold,
        num_workers=4
    )

    backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    config = UperNetConfig(backbone_config=backbone_config, num_labels=number_of_classes)
    model = UperNetForSemanticSegmentation(config)

    # Load checkpoint đã train xong
    model_path = "upernet_convnext_trained_model"
    model.from_pretrained(model_path)
    model.to(device)

    criterion = CrossEntropyLoss(ignore_index=255)

    validate_loop(val_dataloader, model, criterion, device)
