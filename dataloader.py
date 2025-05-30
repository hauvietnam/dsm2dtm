import os
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader

class DSMPatchFolderDataset(Dataset):
    def __init__(self, root_dir, patch_size=256, nodata_val=-9999, nodata_threshold=0.1, transforms=None):
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "masks")
        self.patch_size = patch_size
        self.nodata_val = nodata_val
        self.nodata_threshold = nodata_threshold
        self.transforms = transforms

        images_all = set(os.listdir(self.images_dir))
        masks_all = set(os.listdir(self.masks_dir))

        # Lấy giao file cùng tên giữa images và masks
        common_files = sorted(images_all.intersection(masks_all))
        if len(common_files) == 0:
            raise RuntimeError("Không tìm thấy file ảnh nào có nhãn tương ứng!")

        self.image_files = common_files
        self.mask_files = common_files

        self.patch_index = []
        self._prepare_patches()

    def _prepare_patches(self):
        print("Bắt đầu tạo index patch ...")
        for file_idx, filename in enumerate(self.image_files):
            image_path = os.path.join(self.images_dir, filename)
            with rasterio.open(image_path) as src:
                img = src.read(1)
            H, W = img.shape
            ps = self.patch_size

            # Tính list vị trí i (hàng) cho patch, cho phép overlap patch cuối
            i_positions = []
            i = 0
            while i + ps <= H:
                i_positions.append(i)
                i += ps
            if len(i_positions) == 0 or i_positions[-1] + ps < H:
                i_positions.append(H - ps)

            # Tương tự cho j (cột)
            j_positions = []
            j = 0
            while j + ps <= W:
                j_positions.append(j)
                j += ps
            if len(j_positions) == 0 or j_positions[-1] + ps < W:
                j_positions.append(W - ps)

            for i_pos in i_positions:
                for j_pos in j_positions:
                    patch = img[i_pos:i_pos+ps, j_pos:j_pos+ps]
                    nodata_ratio = np.sum(patch == self.nodata_val) / (ps * ps)
                    if nodata_ratio < self.nodata_threshold:
                        self.patch_index.append((file_idx, i_pos, j_pos))
        print(f"Tạo index patch xong, tổng số patch: {len(self.patch_index)}")

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        file_idx, top, left = self.patch_index[idx]

        img_path = os.path.join(self.images_dir, self.image_files[file_idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[file_idx])

        with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
            img = src_img.read(1)[top:top+self.patch_size, left:left+self.patch_size]
            mask = src_mask.read(1)[top:top+self.patch_size, left:left+self.patch_size]

        img = np.where(img == self.nodata_val, 0, img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).repeat(3,1,1)
        mask_tensor = torch.tensor(mask, dtype=torch.long)

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, mask_tensor

def get_dataloader(root_dir, batch_size=8, patch_size=256, nodata_val=-9999, nodata_threshold=0.1, num_workers=4):
    dataset = DSMPatchFolderDataset(
        root_dir,
        patch_size=patch_size,
        nodata_val=nodata_val,
        nodata_threshold=nodata_threshold
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
def get_train_val_dataloaders(root_dir, batch_size=8, patch_size=256, nodata_val=-9999, nodata_threshold=0.1, val_split=0.2, num_workers=4):
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