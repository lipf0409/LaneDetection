# -*- coding: utf-8 -*-
import datetime, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from models import LaneDetectionModel
from dataset import LaneDetectionDataset
import random


# ---------------- 工具函数 ----------------
def current_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


@torch.no_grad()
def calculate_iou(pred, target, eps=1e-6):
    """修复IoU计算：放宽过滤条件，避免全0返回"""
    # pred: 模型输出经过sigmoid后的结果 (B,1,H,W)
    # target: 二值化后的标签 (B,1,H,W)
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    # 按批次计算每个样本的IoU
    inter = (pred_bin * target_bin).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + target_bin.sum(dim=[1, 2, 3]) - inter

    # 仅过滤union为0的样本（避免过度过滤）
    mask = union > eps
    if mask.sum() == 0:
        return 0.0

    # 只计算有效样本的IoU
    iou_per_sample = (inter[mask] + eps) / (union[mask] + eps)
    return iou_per_sample.mean().item()


@torch.no_grad()
def batched_dice(pred, target, eps=1e-6):
    """修复Dice计算：简化逻辑，避免过滤过严"""
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    inter = (pred_bin * target_bin).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + target_bin.sum(dim=[1, 2, 3])

    # 不过滤样本，仅避免除0
    dice_per_sample = (2. * inter + eps) / (union + eps)
    return dice_per_sample.mean()


def criterion(outputs, targets):
    def focal_loss(logits, targets, alpha=0.95, gamma=2):  # alpha从0.25→0.95，侧重正样本（车道线）
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** gamma
        # 对正负样本分别加权：alpha给正样本，1-alpha给负样本
        alpha_t = torch.where(targets == 1, alpha, 1 - alpha)
        loss = alpha_t * focal_term * bce_loss
        return loss.mean()

    foc = focal_loss(outputs, targets)
    pred = torch.sigmoid(outputs)
    dice = 1 - batched_dice(pred, targets)

    return 0.4 * foc + 0.6 * dice  # 进一步增加Dice权重，直接对齐IoU


def save_metrics(metrics, path):
    """保存训练指标"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ---------------- 主程序 ----------------
if __name__ == '__main__':
    # 强制UTF-8编码
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # 配置参数：优化学习率、正则化、早停等关键参数
    cfg = {
        'batch-size': 32,
        'num-workers': 4,
        'num-epochs': 50,
        'learning-rate': 2e-5,  # 提高学习率，加快收敛（原5e-5过慢）
        'weight-decay': 1e-5,   # 降低权重衰减，避免正则化过强
        'dropout-rate': 0.2,    # 降低dropout，保留更多特征
        'log-interval': 10,
        'load-pretrained': True,
        'load-checkpoint': False,
        'best-checkpoint-path': Path('checkpoints/best-ckpt.pt'),
        'last-checkpoint-path': Path('checkpoints/last-ckpt.pt'),
        'metrics-path': Path('metrics.json'),
        'grad-accumulation-steps': 1,
        'mixed_precision': True,
        'pin_memory': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'patience': 15,
    }

    # 路径配置
    data_root = Path(r'/root/autodl-tmp/kaggle_cache/datasets/manideep1108/tusimple/versions/5/TUSimple')
    train_dir = data_root / 'train_set'
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 数据变换：仅归一化（增强已在dataset中实现）
    train_transform = T.Compose([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = T.Compose([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 修复数据集拆分：确保无重叠 + 可复现
    # 1. 创建独立的训练集和验证集实例
    train_ds_full = LaneDetectionDataset(
        str(train_dir),
        transform=train_transform,
        split='train_set',
        augment=True  # 训练集开启增强
    )

    valid_ds_full = LaneDetectionDataset(
        str(train_dir),
        transform=valid_transform,
        split='train_set',
        augment=False  # 验证集关闭增强
    )

    # 2. 数据集拆分（固定种子，确保无重叠）
    total_samples = len(train_ds_full)
    train_ratio = 0.7
    train_size = int(train_ratio * total_samples)
    valid_size = total_samples - train_size

    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成索引
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    train_indices = all_indices[:train_size]
    valid_indices = all_indices[train_size:]

    # 3. 创建子集
    train_ds = Subset(train_ds_full, train_indices)
    valid_ds = Subset(valid_ds_full, valid_indices)

    # 数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch-size'],
        shuffle=True,
        num_workers=cfg['num-workers'],
        pin_memory=cfg['pin_memory'],
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg['batch-size'],
        shuffle=False,
        num_workers=cfg['num-workers'],
        pin_memory=cfg['pin_memory']
    )

    # 调试：验证数据集标签是否正确
    print("\n=== 数据集调试 ===")
    sample_imgs, sample_lbls = next(iter(train_loader))
    print(f"批次尺寸：图片 {sample_imgs.shape}，标签 {sample_lbls.shape}")
    print(f"标签最小值：{sample_lbls.min()}，最大值：{sample_lbls.max()}")
    lane_pixel_ratio = (sample_lbls == 1).sum() / sample_lbls.numel()
    print(f"批次车道线像素占比：{lane_pixel_ratio:.4f}")
    assert lane_pixel_ratio > 0, "标签加载错误：无车道线像素！"

    # 模型、优化器和调度器
    model = LaneDetectionModel(
        pretrained=cfg['load-pretrained'],
        dropout_rate=cfg['dropout-rate']
    ).to(cfg['device'])

    # 优化器：AdamW + 合理学习率
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['learning-rate'],
        weight_decay=cfg['weight-decay']
    )

    # 学习率调度器：缩短周期，加快收敛

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 缓慢衰减

    # 加载检查点
    start_epoch = 0
    metrics = {
        'train_loss': [], 'valid_loss': [],
        'train_iou': [], 'valid_iou': []
    }
    best_val_iou = 0.0
    no_improve_epochs = 0

    if cfg['load-checkpoint'] and cfg['best-checkpoint-path'].exists():
        checkpoint = torch.load(cfg['best-checkpoint-path'], map_location=cfg['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_iou = checkpoint['best_val_iou']

        if cfg['metrics-path'].exists():
            with open(cfg['metrics-path'], 'r') as f:
                metrics = json.load(f)
        print(f"{current_time()} 已加载检查点，从第 {start_epoch} 轮开始训练")

    # 混合精度训练
    scaler = None
    if cfg['mixed_precision'] and cfg['device'] == 'cuda':
        scaler = torch.amp.GradScaler()

    # 验证模型输出尺寸
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 736, 1280).to(cfg['device'])
        out = model(dummy)
        print(f'模型输出尺寸: {out.shape} (预期: (1, 1, 736, 1280))')
        assert out.shape == (1, 1, 736, 1280), "模型输出尺寸错误"

    # 打印数据集信息
    print(f"\n数据集拆分：训练集 {len(train_ds)} 样本 | 验证集 {len(valid_ds)} 样本")
    print('---------- 训练开始 (防过拟合优化版) ----------\n')

    # 训练循环
    for epoch in range(start_epoch, cfg['num-epochs']):
        # ---- 训练阶段 ----
        model.train()
        train_total_loss = 0.0
        train_total_iou = 0.0
        batch_count = 0

        for batch_idx, (imgs, lbls) in enumerate(train_loader, 1):
            imgs, lbls = imgs.to(cfg['device']), lbls.to(cfg['device'])
            optimizer.zero_grad()

            # 混合精度前向传播
            if cfg['mixed_precision'] and cfg['device'] == 'cuda':
                with torch.amp.autocast('cuda', enabled=True):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            # 反向传播 + 梯度裁剪
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 计算IoU
            with torch.no_grad():
                pred = torch.sigmoid(outputs)
                iou = calculate_iou(pred, lbls)

            # 累加指标
            train_total_loss += loss.item()
            train_total_iou += iou
            batch_count += 1

            # 打印日志
            if batch_idx % cfg['log-interval'] == 0:
                avg_batch_loss = train_total_loss / batch_count
                avg_batch_iou = train_total_iou / batch_count
                print(f'{current_time()} [训练] '
                      f'轮次:{epoch:03d} 批次:{batch_idx:04d}/{len(train_loader):04d} '
                      f'损失值:{avg_batch_loss:.5f} IoU:{avg_batch_iou:.5f}')

        # 训练集本轮指标
        avg_train_loss = train_total_loss / len(train_loader)
        avg_train_iou = train_total_iou / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        metrics['train_iou'].append(avg_train_iou)

        # ---- 验证阶段 ----
        model.eval()
        valid_total_loss = 0.0
        valid_total_iou = 0.0

        with torch.no_grad():
            for imgs, lbls in valid_loader:
                imgs, lbls = imgs.to(cfg['device']), lbls.to(cfg['device'])

                if cfg['mixed_precision'] and cfg['device'] == 'cuda':
                    with torch.amp.autocast('cuda', enabled=True):
                        outputs = model(imgs)
                        loss = criterion(outputs, lbls)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)

                pred = torch.sigmoid(outputs)
                iou = calculate_iou(pred, lbls)

                valid_total_loss += loss.item()
                valid_total_iou += iou

        # 验证集本轮指标
        avg_valid_loss = valid_total_loss / len(valid_loader)
        avg_valid_iou = valid_total_iou / len(valid_loader)
        metrics['valid_loss'].append(avg_valid_loss)
        metrics['valid_iou'].append(avg_valid_iou)

        # 学习率调度
        scheduler.step()

        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_iou': best_val_iou
        }

        torch.save(checkpoint, cfg['last-checkpoint-path'])

        # 保存最优模型（提升阈值0.001）
        if avg_valid_iou > best_val_iou + 0.001:
            best_val_iou = avg_valid_iou
            torch.save(checkpoint, cfg['best-checkpoint-path'])
            print(f'{current_time()} [模型保存] 验证集IoU提升至 {best_val_iou:.5f}，已保存最优模型！')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f'{current_time()} [早停监测] 已 {no_improve_epochs} 轮未提升')

        # 早停机制
        if no_improve_epochs >= cfg['patience']:
            print(f'{current_time()} [早停触发] 验证集性能 {cfg["patience"]} 轮未提升，提前终止训练')
            break

        # 保存指标
        save_metrics(metrics, cfg['metrics-path'])

        # 打印轮次总结
        print(f'{current_time()} [总结] 轮次:{epoch:03d} 完成 | '
              f'训练损失:{avg_train_loss:.5f} 训练IoU:{avg_train_iou:.5f} | '
              f'验证损失:{avg_valid_loss:.5f} 验证IoU:{avg_valid_iou:.5f}\n')

    print(f'\n---------- 训练完成 – 最优验证IoU: {best_val_iou:.5f} ----------\n')
