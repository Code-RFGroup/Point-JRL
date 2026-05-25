import torch
import math
import shutil
import yaml
import os
import random
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import json
from datetime import datetime
from pathlib import Path
from torch_geometric.loader import DataLoader as GNNDataLoader
from models import simsiam, pointnext, hybrid, point_simsiam, lpa, gnn
from dataset.dataset import PointDataset, ImageDataset, LPADataset, GraphDataset
from dataset.augmentation import TwoCropsTransform, get_layout_transforms, get_point_transforms, get_lpa_transforms
from utils.criterion import SimSiamLoss, HybridLoss, LPALoss, GNNLoss


# ------------------------------------- Universal functions -------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def increment_dir(path_str):
    """Create an auto-incremented directory (e.g., outputs/run -> outputs/run_1 -> ...)."""
    path = Path(path_str)
    base = path
    counter = 1
    while path.exists():
        path = base.parent / f"{base.name}_{counter}"
        counter += 1
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def load_config(config_path):
    """Load YAML config and flatten model-specific section."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_name = config['model']
    model_config = config['models'][model_name]
    model_config['global'] = {
        'description': config['description'],
        'gpu_ids': config['gpu_ids'],
        'seed': config['seed'],
        'model_name': model_name
    }

    return model_config


def build_model(config):
    """Build model based on config."""
    model_name = config['global']['model_name']

    if model_name == 'simsiam':
        arch_cfg = config['arch']
        print(f"=> Creating SimSiam with backbone: {arch_cfg['name']}")
        base_encoder = models.__dict__[arch_cfg['name']]
        model = simsiam.SimSiam(base_encoder, arch_cfg['dim'], arch_cfg['pred_dim'])

    elif model_name == 'pointnext':
        arch_cfg = config['arch']
        print(f"=> Creating PointNeXt SimSiam: {arch_cfg['name']}")
        base_encoder = pointnext.PyGPointNextEncoder(
            in_channels=arch_cfg['in_channels'],
            width=arch_cfg['width'],
            k=arch_cfg['k'],
            latent_dim=arch_cfg['latent_dim']
        )
        model = point_simsiam.PointSimSiam(
            base_encoder=base_encoder,
            encoder_out_dim=arch_cfg['latent_dim'],
            dim=arch_cfg['dim'],
            pred_dim=arch_cfg['pred_dim']
        )

    elif model_name == 'hybrid_point':
        arch_cfg = config['arch']
        print(f"=> Creating Hybrid PointNeXt (SimSiam + Cls): {arch_cfg['name']}")
        base_encoder = pointnext.PyGPointNextEncoder(
            in_channels=arch_cfg['in_channels'],
            width=arch_cfg['width'],
            k=arch_cfg['k'],
            latent_dim=arch_cfg['latent_dim']
        )
        num_classes = arch_cfg['num_classes']
        model = hybrid.HybridSimSiam(
            base_encoder=base_encoder,
            encoder_out_dim=arch_cfg['latent_dim'],
            dim=arch_cfg['dim'],
            pred_dim=arch_cfg['pred_dim'],
            num_classes=num_classes
        )

    elif model_name == 'hybrid_image':
        arch_cfg = config['arch']
        print(f"=> Creating Hybrid ResNet (SimSiam + Cls): {arch_cfg['name']}")
        base_encoder = models.__dict__[arch_cfg['name']](pretrained=False)
        base_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        prev_dim = base_encoder.fc.weight.shape[1]
        base_encoder.fc = nn.Identity()
        num_classes = arch_cfg['num_classes']
        model = hybrid.HybridSimSiam(
            base_encoder=base_encoder,
            encoder_out_dim=prev_dim,
            dim=arch_cfg['dim'],
            pred_dim=arch_cfg['pred_dim'],
            num_classes=num_classes,
        )

    elif model_name == 'lpa':
        model = lpa.LPA()

    elif model_name == 'gnn':
        model = gnn.DATE25_GNN()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def build_dataloader(config):
    """Build train/val/test dataloaders based on config."""
    model_name = config['global']['model_name']
    data_cfg = config['data']

    data_root = data_cfg['root']
    split_root = data_cfg.get('split_root', None)
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg['num_workers']

    LoaderClass = GNNDataLoader if model_name == 'gnn' else DataLoader

    train_dataset = None
    transform = None
    if model_name in ['simsiam', 'hybrid_image']:
        transform = TwoCropsTransform(get_layout_transforms())
        TargetDataset = ImageDataset
    elif model_name in ['pointnext', 'hybrid_point']:
        transform = TwoCropsTransform(get_point_transforms(config['arch']['in_channels']))
        TargetDataset = PointDataset
    elif model_name == 'lpa':
        transform = get_lpa_transforms()
        TargetDataset = LPADataset
    elif model_name == 'gnn':
        transform = None
        TargetDataset = GraphDataset

    train_split = os.path.join(split_root, 'train.csv') if split_root else None
    train_dataset = TargetDataset(data_root, train_split, transform=transform)

    val_dataset, test_dataset = None, None
    if split_root:
        val_dataset = TargetDataset(data_root, os.path.join(split_root, 'val.csv'), transform=None)
        test_dataset = TargetDataset(data_root, os.path.join(split_root, 'test.csv'), transform=None)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config['global']['seed'])

    if model_name in ['simsiam', 'lpa', 'pointnext']:
        print(f"=> [Dataloader] Using natural distribution (shuffle) for {model_name}")
        sampler = None
        shuffle = True
    else:
        print(f"=> [Dataloader] Using weighted resampling for {model_name}")
        train_targets = np.array(train_dataset.labels)
        class_counts = np.bincount(train_targets)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        samples_weight = torch.from_numpy(class_weights[train_targets]).double()

        sampler = WeightedRandomSampler(
            weights=samples_weight,
            num_samples=len(samples_weight),
            replacement=True,
            generator=g
        )
        shuffle = False

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": g
    }

    train_loader = LoaderClass(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        **common_kwargs
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = LoaderClass(val_dataset, shuffle=False, drop_last=False, **common_kwargs)

    test_loader = None
    if test_dataset is not None:
        test_loader = LoaderClass(test_dataset, shuffle=False, drop_last=False, **common_kwargs)

    return train_loader, val_loader, test_loader


# ------------------------------------- Training functions -------------------------------------

def adjust_lr(optimizer, init_lr, epoch, total_epochs, warmup_epochs=10, min_lr=1e-6):
    """Cosine annealing LR schedule with linear warmup."""
    if epoch < warmup_epochs:
        cur_lr = init_lr * ((epoch + 1) / warmup_epochs)
    else:
        curr_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        cur_lr = min_lr + (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * curr_progress))

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

    return cur_lr


def build_optimizer(model, config):
    """Build optimizer with optional fixed predictor LR."""
    model_name = config['global']['model_name']
    train_cfg = config['train']

    optim_name = train_cfg['optimizer']

    if train_cfg.get('fix_pred_lr', False) and hasattr(model, 'predictor'):
        print(f"=> [Optimizer] Fixing Predictor LR for {model_name}")
        predictor_ids = list(map(id, model.predictor.parameters()))
        base_params = filter(lambda p: id(p) not in predictor_ids, model.parameters())

        optim_params = [
            {'params': base_params, 'fix_lr': False},
            {'params': model.predictor.parameters(), 'fix_lr': True}
        ]
    else:
        optim_params = model.parameters()

    base_lr = train_cfg['init_lr']

    if model_name == 'simsiam':
        init_lr = base_lr * (config['data']['batch_size'] / 256)
    else:
        init_lr = base_lr

    weight_decay = train_cfg['weight_decay']

    if optim_name == 'adamw':
        print(f"=> Using AdamW for {model_name} (lr={init_lr}, weight_decay={weight_decay})")
        optimizer = torch.optim.AdamW(optim_params, lr=init_lr, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        print(f"=> Using SGD for {model_name} (lr={init_lr}, momentum={train_cfg['momentum']}, weight_decay={weight_decay})")
        optimizer = torch.optim.SGD(optim_params, lr=init_lr,
                                    momentum=train_cfg['momentum'],
                                    weight_decay=weight_decay)
    elif optim_name == 'adam':
        print(f"=> Using Adam for {model_name} (lr={init_lr}, weight_decay={weight_decay})")
        optimizer = torch.optim.Adam(optim_params, lr=init_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

    return optimizer, init_lr


def build_criterion(config):
    """Build loss criterion based on model type."""
    model_name = config['global']['model_name']

    if model_name in ['hybrid_point', 'hybrid_image']:
        criterion = HybridLoss(lambda_cls=config['train']['lambda_cls'])
    elif model_name == 'simsiam':
        criterion = SimSiamLoss()
    elif model_name == 'pointnext':
        criterion = SimSiamLoss()
    elif model_name == 'lpa':
        criterion = LPALoss()
    elif model_name == 'gnn':
        criterion = GNNLoss()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return criterion


def train_log(save_dir, log_info):
    """Append training log entry to JSONL file."""
    log_info['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = os.path.join(save_dir, 'train_log.jsonl')

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_info, indent=4) + '\n')


def checkpoint(state, is_best, save_dir):
    """Save model checkpoint. Always saves last.pth; copies to best.pth if best."""
    filename = 'last.pth'
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(save_dir, 'best.pth')
        shutil.copyfile(filepath, best_filepath)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
