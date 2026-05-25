import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from natsort import natsorted
import random


class PointDataset(Dataset):
    """Point cloud dataset.
    Supports 4D data (x, y, nx, ny) and legacy 2D data (x, y) with auto-padding.
    Reads train/val/test splits from CSV files.
    Label convention: NHS=0, HS=1.
    """
    def __init__(self, data_root, split_file, transform):
        self.data_root = data_root
        self.transform = transform
        self.file_ids = []
        self.labels = []

        if split_file is not None:
            print(f"=> Loading split from {split_file}")
            df = pd.read_csv(split_file)
            self.file_ids = df['filename'].astype(str).tolist()
            self.labels = df['label'].tolist()
        else:
            print(f"=> Scanning directory: {data_root}")
            all_files = natsorted([f for f in os.listdir(data_root) if f.endswith('.npy')])
            self.file_ids = [os.path.splitext(f)[0] for f in all_files]
            self.labels = [self._parse_label(f) for f in self.file_ids]

    def _parse_label(self, filename):
        """NHS -> 0, HS -> 1"""
        if 'NHS' in filename:
            return 0
        elif 'HS' in filename:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]

        filename = f"{file_id}.npy"
        file_path = os.path.join(self.data_root, filename)
        pts = np.load(file_path).astype(np.float32)  # [N, 2] or [N, 4]

        # Auto-pad 2D data to 3D
        if pts.shape[1] == 2:
            z = np.zeros((pts.shape[0], 1), dtype=pts.dtype)
            pts = np.hstack([pts, z])  # [N, 3]

        pts = torch.from_numpy(pts).float().transpose(1, 0).contiguous()

        if self.transform is not None:
            views = self.transform(pts)
            return views, label, filename
        else:
            return pts, label, filename


class ImageDataset(Dataset):
    """Image dataset for SimSiam experiments.
    Reads grayscale PNG images with CSV split support.
    """
    def __init__(self, data_root, split_file, transform):
        self.data_root = data_root
        self.transform = transform
        self.file_ids = []
        self.labels = []

        if split_file is not None:
            df = pd.read_csv(split_file)
            self.file_ids = df['filename'].astype(str).tolist()
            self.labels = df['label'].tolist()
        else:
            all_files = natsorted([f for f in os.listdir(data_root) if f.endswith('.png')])
            self.file_ids = [os.path.splitext(f)[0] for f in all_files]
            self.labels = [self._parse_label(f) for f in self.file_ids]

    def _parse_label(self, filename):
        if 'NHS' in filename:
            return 0
        elif 'HS' in filename:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]

        filename = f"{file_id}.png"
        file_path = os.path.join(self.data_root, filename)

        img = Image.open(file_path).convert('L')
        img = TF.to_tensor(img)  # [1, H, W]
        img = TF.normalize(img, mean=[0.5], std=[0.5])

        if self.transform is not None:
            views = self.transform(img)
            return views, label, filename
        else:
            return img, label, filename


class LPADataset(Dataset):
    """Layout Pattern Analysis dataset with triplet sampling.
    Returns (anchor, positive, negative) tuples for contrastive learning.
    """
    def __init__(self, data_root, split_file, transform):
        self.data_root = data_root
        self.transform = transform
        self.file_ids = []
        self.labels = []

        if split_file is not None:
            df = pd.read_csv(split_file)
            self.file_ids = df['filename'].astype(str).tolist()
            self.labels = df['label'].tolist()
        else:
            all_files = natsorted([f for f in os.listdir(data_root) if f.endswith('.png')])
            self.file_ids = [os.path.splitext(f)[0] for f in all_files]
            self.labels = [self._parse_label(f) for f in self.file_ids]

    def _parse_label(self, filename):
        if 'NHS' in filename:
            return 0
        elif 'HS' in filename:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]
        filename = f"{file_id}.png"
        file_path = os.path.join(self.data_root, filename)

        img = Image.open(file_path).convert('L')
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5], std=[0.5])

        if self.transform is not None:
            positive = self.transform(img)

            neg_idx = idx
            while neg_idx == idx:
                neg_idx = random.randint(0, len(self.file_ids) - 1)

            neg_file_id = self.file_ids[neg_idx]
            neg_filename = f"{neg_file_id}.png"
            neg_file_path = os.path.join(self.data_root, neg_filename)

            neg_img = Image.open(neg_file_path).convert('L')
            negative_tensor = TF.to_tensor(neg_img)
            negative_tensor = TF.normalize(negative_tensor, mean=[0.5], std=[0.5])

            return [img, positive, negative_tensor], label, filename

        else:
            return img, label, filename


class GraphDataset(Dataset):
    """Graph dataset for GNN experiments.
    Reads PyTorch Geometric Data objects (.pt files) with CSV split support.
    """
    def __init__(self, data_root, split_file, transform):
        self.data_root = data_root
        self.transform = transform
        self.file_ids = []
        self.labels = []

        if split_file is not None:
            print(f"=> Loading split from {split_file}")
            df = pd.read_csv(split_file)
            self.file_ids = df['filename'].astype(str).tolist()
            self.labels = df['label'].tolist()
        else:
            print(f"=> Scanning directory: {data_root}")
            all_files = natsorted([f for f in os.listdir(data_root) if f.endswith('.pt')])
            self.file_ids = [os.path.splitext(f)[0] for f in all_files]
            self.labels = [self._parse_label(f) for f in self.file_ids]

    def _parse_label(self, filename):
        """NHS -> 0, HS -> 1"""
        if 'NHS' in filename:
            return 0
        elif 'HS' in filename:
            return 1
        else:
            return 0

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]

        filename = f"{file_id}.pt"
        file_path = os.path.join(self.data_root, filename)

        data = torch.load(file_path, weights_only=False)

        # Force-overwrite data.y to ensure consistency with CSV labels
        data.y = torch.tensor([label], dtype=torch.long)

        return data, label, filename
