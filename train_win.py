# train_win.py  – 完整可运行版（Windows / 4050 显存）
import datetime, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models import LaneDetectionModel
from dataset import LaneDetectionDataset


# ---------------- 工具函数 ----------------
def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@torch.no_grad()
def calculate_iou(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


@torch.no_grad()
def batched_dice(pred, target, eps=1e-6):
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return (2. * inter + eps) / (union + eps)


def criterion(outputs, targets):
    outputs = outputs.flatten(1)
    targets = targets.flatten(1)
    bce = nn.functional.binary_cross_entropy(outputs, targets)
    dice = 1 - batched_dice(outputs, targets).mean()
    return bce + dice


# ---------------- 主程序 ----------------
if __name__ == '__main__':
    cfg = {
        'num-epochs': 100,
        'batch-size': 2,          # 4050 笔记本显存
        'num-workers': 2,         # Windows spawn 安全值
        'learning-rate': 5e-4,
        'weight-decay': 1e-4,
        'log-interval': 10,
        'load-pretrained': True,
        'load-checkpoint': False,  # 第一次训练设为 False
        'best-checkpoint-path': Path('checkpoints/best-ckpt1.pt'),
        'last-checkpoint-path': Path('checkpoints/last-ckpt1.pt'),
    }

    # 路径
    data_root = Path(r'E:\PY_CODE\LaneDetection-main\datasets\TUSimple')
    train_dir = data_root / ''
    valid_dir = data_root / ''   # 如有 validation 文件夹可换

    # 变换（Dataset 内部已做 resize 到 32 倍数 + 归一化）
    train_tf = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    valid_tf = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_ds = LaneDetectionDataset(str(train_dir), transform=train_tf, split='train_set')
    valid_ds = LaneDetectionDataset(str(valid_dir), transform=valid_tf, split='test_set')

    train_loader = DataLoader(train_ds, batch_size=cfg['batch-size'],
                              shuffle=True, num_workers=cfg['num-workers'], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['batch-size'],
                              shuffle=False, num_workers=cfg['num-workers'], pin_memory=True)

    # 模型与优化器
    model = LaneDetectionModel(pretrained=cfg['load-pretrained']).cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning-rate'], weight_decay=cfg['weight-decay'])

    # 第一次训练先不加载旧权重；后续改成 True 即可
    # if cfg['load-checkpoint'] and cfg['best-checkpoint-path'].exists():
    #     model.load_state_dict(torch.load(cfg['best-checkpoint-path'], map_location='cuda'))

    best_iou = 0.0
    log_int = cfg['log-interval']
    print('\n---------- training start (Windows) ----------\n')

    for epoch in range(cfg['num-epochs']):
        # ---- train ----
        model.train()
        for batch_idx, (imgs, lbls) in enumerate(train_loader, 1):
            imgs, lbls = imgs.cuda(), lbls.cuda()
            optimizer.zero_grad()
            loss = criterion(model(imgs).sigmoid(), lbls)
            loss.backward()
            optimizer.step()

            if batch_idx % log_int == 0:
                print(f'{current_time()} [train] '
                      f'{epoch:03d}/{batch_idx:04d}/{len(train_loader):04d}] '
                      f'loss: {loss.item():.5f}')

        # ---- validate ----
        model.eval()
        all_pred, all_true = [], []
        for imgs, lbls in valid_loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            pred = model(imgs).sigmoid().flatten()
            all_pred.append(pred.cpu())
            all_true.append(lbls.cpu().flatten())
        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)
        iou = calculate_iou((all_pred > 0.5).int(), all_true).item()

        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), cfg['best-checkpoint-path'])
        torch.save(model.state_dict(), cfg['last-checkpoint-path'])
        print(f'{current_time()} [valid] {epoch:03d}] IoU: {iou:.4f}   best: {best_iou:.4f}')

    print(f'\n---------- training finished – best IoU: {best_iou:.3f} ----------\n')