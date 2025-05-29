import torch
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataloader import get_dataloader  # import hàm get_dataloader từ file dataloader.py

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

if __name__ == "__main__":
    dataset_root = "/kaggle/input/dsm2dtm/datasets"  # chỉ cần folder data, ví dụ "./dataset"
    batch_size = 8
    patch_size = 256
    num_epochs = 10
    nodata_val = -9999
    nodata_threshold = 0.1
    number_of_classes = 2  # chỉnh số lớp theo dữ liệu

    # Kiểm tra thiết bị, ưu tiên CUDA nếu có
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"CUDA current device index: {torch.cuda.current_device()}")

    dataloader = get_dataloader(
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
    model.to(device)  # Chuyển model lên GPU hoặc CPU

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loop(dataloader, model, criterion, optimizer, device)
        save_path = f"upernet_convnext_epoch_{epoch+1}"
        model.save_pretrained(save_path)
        print(f"Model đã được lưu tại: {save_path}")
