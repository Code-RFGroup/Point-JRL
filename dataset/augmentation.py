import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class TwoCropsTransform:
    """Generate two augmented views of the same input."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class RandomRotate90:
    """Random rotation by 0, 90, 180, or 270 degrees."""
    def __call__(self, x):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(x, angle, interpolation=InterpolationMode.NEAREST)


def get_layout_transforms():
    """Data augmentation for layout images (SimSiam / Hybrid Image)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        transforms.RandomAffine(degrees=0, translate=(0.01, 0.01),
                                interpolation=InterpolationMode.NEAREST, fill=0)
    ])


def get_lpa_transforms():
    """Data augmentation for LPA triplet training (larger translation range)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandomRotate90(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                interpolation=InterpolationMode.NEAREST, fill=0)
    ])


class PointAugment_4d:
    """Data augmentation for 4D point clouds (x, y, nx, ny).
    Applies random 90-degree rotations, flips, and small translations.
    Normal vectors are transformed consistently with coordinates.
    """
    def __init__(self, shift_range=0.005, flip_prob=0.5):
        self.shift_range = shift_range
        self.flip_prob = flip_prob

    def __call__(self, points):
        # Input: [4, N] -> (x, y, nx, ny)
        points = points.clone()
        points[0:2, :] -= 0.5

        x = points[0, :].clone()
        y = points[1, :].clone()
        nx = points[2, :].clone()
        ny = points[3, :].clone()

        k = random.randint(0, 3)
        if k > 0:
            if k == 1:   # 90 deg: (x, y) -> (-y, x)
                new_x, new_y = -y, x
                new_nx, new_ny = -ny, nx
            elif k == 2: # 180 deg: (x, y) -> (-x, -y)
                new_x, new_y = -x, -y
                new_nx, new_ny = -nx, -ny
            elif k == 3: # 270 deg: (x, y) -> (y, -x)
                new_x, new_y = y, -x
                new_nx, new_ny = ny, -nx
            else:
                new_x, new_y = x, y
                new_nx, new_ny = nx, ny

            points[0, :] = new_x
            points[1, :] = new_y
            points[2, :] = new_nx
            points[3, :] = new_ny

        if random.random() < self.flip_prob:
            points[0, :] = -points[0, :]  # Horizontal flip
            points[2, :] = -points[2, :]  # Normal X flip

        if random.random() < self.flip_prob:
            points[1, :] = -points[1, :]  # Vertical flip
            points[3, :] = -points[3, :]  # Normal Y flip

        points[0:2, :] += 0.5

        shifts = (torch.rand(2, 1, device=points.device) - 0.5) * 2 * self.shift_range
        points[0:2, :] += shifts

        return points


class PointAugment:
    """Data augmentation for 2D/3D point clouds.
    Applies random 90-degree rotations, flips, and small translations.
    """
    def __init__(self, shift_range=0.005, flip_prob=0.5):
        self.shift_range = shift_range
        self.flip_prob = flip_prob

    def __call__(self, points):
        points = points.clone()

        # Center around origin for rotation
        points[0:2, :] -= 0.5

        x = points[0, :].clone()
        y = points[1, :].clone()

        k = random.randint(0, 3)
        if k > 0:
            if k == 1:   # 90 deg
                new_x, new_y = -y, x
            elif k == 2: # 180 deg
                new_x, new_y = -x, -y
            elif k == 3: # 270 deg
                new_x, new_y = y, -x
            else:
                new_x, new_y = x, y

            points[0, :] = new_x
            points[1, :] = new_y

        if random.random() < self.flip_prob:
            points[0, :] = -points[0, :]

        if random.random() < self.flip_prob:
            points[1, :] = -points[1, :]

        # Move back
        points[0:2, :] += 0.5

        shifts = (torch.rand(3, 1, device=points.device) - 0.5) * 2 * self.shift_range
        shifts[2, :] = 0
        points += shifts

        return points


def get_point_transforms(dims=4):
    """Get point cloud augmentation transforms.
    Args:
        dims: 4 for (x, y, nx, ny) data, otherwise 2D/3D
    """
    if dims == 4:
        return transforms.Compose([PointAugment_4d()])
    else:
        return transforms.Compose([PointAugment()])
