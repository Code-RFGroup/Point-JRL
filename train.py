import os
import yaml
import torch
from tqdm import tqdm
from utils import utils
from torch_geometric.nn import DataParallel as GNNDataParallel


def train_epoch_ssl(loader, model, criterion, optimizer, scaler, epoch, config, device):
    """Self-supervised learning epoch (SimSiam / PointNeXt)."""
    model.train()
    losses = utils.AverageMeter('Loss', ':.4f')

    with tqdm(loader, desc=f'Train Ep {epoch} [SSL]', leave=False) as pbar:
        for inputs, _, _ in pbar:
            view1 = inputs[0].to(device, non_blocking=True)
            view2 = inputs[1].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                p1, p2, z1, z2 = model(view1, view2)
                loss = criterion(p1, p2, z1, z2)

            losses.update(loss.item(), view1.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    return {'loss': losses.avg}


def train_epoch_ssl_neg(loader, model, criterion, optimizer, scaler, epoch, config, device):
    """Triplet contrastive learning epoch (LPA)."""
    model.train()
    losses = utils.AverageMeter('Loss', ':.4f')

    with tqdm(loader, desc=f'Train Ep {epoch} [LPA-Triplet]', leave=False) as pbar:
        for inputs, _, _ in pbar:
            img_a = inputs[0].to(device, non_blocking=True)
            img_p = inputs[1].to(device, non_blocking=True)
            img_n = inputs[2].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                z_a = model(img_a)
                z_p = model(img_p)
                z_n = model(img_n)
                loss = criterion(z_a, z_p, z_n)

            losses.update(loss.item(), img_a.size(0))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    return {'loss': losses.avg}


def train_epoch_hybrid(loader, model, criterion, optimizer, scaler, epoch, config, device):
    """Hybrid training epoch (SimSiam SSL + Classification)."""
    model.train()

    losses_total = utils.AverageMeter('Total', ':.4f')
    losses_ssl = utils.AverageMeter('SSL', ':.4f')
    losses_cls = utils.AverageMeter('CLS', ':.4f')

    with tqdm(loader, desc=f'Train Ep {epoch} [Hybrid]', leave=False) as pbar:
        for inputs, labels, _ in pbar:
            view1 = inputs[0].to(device, non_blocking=True)
            view2 = inputs[1].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                p1, p2, z1, z2, logits1, logits2 = model(view1, view2)
                if epoch < config['train']['ssl_epoch']:
                    criterion.lambda_cls = 0
                else:
                    criterion.lambda_cls = config['train']['lambda_cls']
                loss, loss_dict = criterion(p1, p2, z1, z2, logits1, logits2, labels)

            bs = view1.size(0)
            losses_total.update(loss.item(), bs)
            losses_ssl.update(loss_dict.get('ssl', 0), bs)
            losses_cls.update(loss_dict.get('cls', 0), bs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({
                'Tot': f'{losses_total.avg:.4f}',
                'SSL': f'{losses_ssl.avg:.4f}',
                'CLS': f'{losses_cls.avg:.4f}'
            })

    return {
        'loss': losses_total.avg,
        'loss_ssl': losses_ssl.avg,
        'loss_cls': losses_cls.avg
    }


def train_epoch_gnn(loader, model, criterion, optimizer, scaler, epoch, config, device):
    """GNN supervised training epoch."""
    model.train()
    losses = utils.AverageMeter('Loss', ':.4f')

    with tqdm(loader, desc=f'Train Ep {epoch} [GNN]', leave=False) as pbar:
        for data, labels, _ in pbar:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits, _ = model(data)
                loss = criterion(logits, labels)

            losses.update(loss.item(), data.num_graphs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    return {'loss': losses.avg}


TRAINERS = {
    'simsiam': train_epoch_ssl,
    'pointnext': train_epoch_ssl,
    'lpa': train_epoch_ssl_neg,
    'hybrid_point': train_epoch_hybrid,
    'hybrid_image': train_epoch_hybrid,
    'gnn': train_epoch_gnn
}


@torch.no_grad()
def validate_epoch(loader, model, criterion, config, device, epoch):
    """Validation epoch (only for hybrid models)."""
    model_name = config['global']['model_name']
    if model_name not in ['hybrid_point', 'hybrid_image']:
        return None
    model.eval()
    cls_losses = utils.AverageMeter('Val CLS Loss', ':.4f')
    acc_meter = utils.AverageMeter('Val Acc', ':.2f')

    with tqdm(loader, desc=f'Valid Ep {epoch}', leave=False) as pbar:
        for inputs, labels, _ in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                p1, p2, z1, z2, logits1, logits2 = model(inputs, inputs)
                _, loss_dict = criterion(p1, p2, z1, z2, logits1, logits2, labels)
                cls_loss_val = loss_dict['cls']
                preds = logits1.argmax(dim=1)
                acc = (preds == labels).float().mean() * 100.0

                cls_losses.update(cls_loss_val.item(), inputs.size(0))
                acc_meter.update(acc.item(), inputs.size(0))

            pbar.set_postfix({
                'Loss': f'{cls_losses.avg:.4f}',
                'Acc': f'{acc_meter.avg:.2f}%'
            })

    return {
        'val_loss_cls': cls_losses.avg,
        'val_acc': acc_meter.avg
    }


def main(config):
    model_name = config['global']['model_name']

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config['global']['gpu_ids']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_parallel = torch.cuda.device_count() > 1

    save_dir = utils.increment_dir(config['train']['save_dir'])
    with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

    train_loader, val_loader, test_loader = utils.build_dataloader(config)
    model = utils.build_model(config).to(device)
    optimizer, init_lr = utils.build_optimizer(model, config)
    criterion = utils.build_criterion(config).to(device)
    scaler = torch.amp.GradScaler('cuda')
    if is_parallel:
        model = torch.nn.DataParallel(model)

    train_one_epoch = TRAINERS[model_name]

    best_metric = float('inf')

    if model_name in ['hybrid_point', 'hybrid_image']:
        metric_name = 'hybrid_score'
    else:
        metric_name = 'train_loss'

    epoch_pbar = tqdm(range(config['train']['epochs']), desc='Global Progress', position=0)
    for epoch in epoch_pbar:
        utils.adjust_lr(optimizer, init_lr, epoch, config['train']['epochs'],
                        config['train']['warm_up'], config['train']['min_lr'])
        cur_lr = optimizer.param_groups[0]['lr']

        train_metrics = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, config, device)

        val_metrics = {}
        val_result = validate_epoch(val_loader, model, criterion, config, device, epoch)
        val_metrics = val_result if val_result is not None else {}

        display_dict = {'lr': f'{cur_lr:.5f}'}
        if model_name in ['hybrid_point', 'hybrid_image']:
            display_dict['tr_ssl'] = f"{train_metrics.get('loss_ssl', 0):.4f}"
            display_dict['val_cls'] = f"{val_metrics['val_loss_cls']:.4f}"
            display_dict['val_acc'] = f"{val_metrics['val_acc']:.2f}%"
        else:
            display_dict['loss'] = f"{train_metrics['loss']:.4f}"

        epoch_pbar.set_postfix(display_dict)

        is_best = False
        score = None
        if model_name in ['hybrid_point', 'hybrid_image']:
            lambda_cls = config['train']['lambda_cls']
            train_ssl = train_metrics['loss_ssl']
            val_cls = val_metrics['val_loss_cls']
            score = train_ssl + lambda_cls * val_cls
        else:
            score = train_metrics['loss']

        if score < best_metric:
            best_metric = score
            is_best = True

        real_model = model.module if is_parallel else model
        state = {
            'epoch': epoch + 1,
            'arch': config['arch'].get('name', None),
            'state_dict': real_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_metric': best_metric,
            'config': config
        }

        utils.checkpoint(state, is_best, save_dir)

        log_stats = {
            'epoch': epoch + 1,
            'lr': cur_lr,
            'metric': score,
            'best_metric': best_metric,
            **train_metrics,
            **val_metrics
        }
        utils.train_log(save_dir, log_stats)

        if is_best:
            msg = f"Epoch {epoch}: New Best {metric_name}: {best_metric:.4f}"
            if model_name in ['hybrid_point', 'hybrid_image']:
                msg += f" (Tr_SSL: {train_metrics['loss_ssl']:.4f} + Val_CLS: {val_metrics['val_loss_cls']:.4f})"
            tqdm.write(msg)


if __name__ == "__main__":
    config_path = './config.yaml'
    config = utils.load_config(config_path)
    utils.set_seed(config['global']['seed'])

    main(config)
