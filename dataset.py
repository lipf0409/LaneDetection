import json, os, random
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import numpy as np


class LaneDetectionDataset(Dataset):
    def __init__(self, root: str, transform=None, split='train_set', augment=False):
        self.root = root
        self.transform = transform
        self.split = split
        self.augment = augment
        self.images = []
        self.labels = []
        self.fixed_size = (1280, 736)  # (width, height) 适配PIL
        self._load_annotations()
        # 确保数据集不为空
        assert len(self.images) > 0, f"在 {root} 中未找到有效数据"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像和标签
        img_path = os.path.join(self.root, self.images[idx])
        lbl_path = os.path.join(self.root, self.labels[idx])

        try:
            image = Image.open(img_path).convert('RGB')
            # 修复：TUSimple标签是彩色图，不转灰度
            label = Image.open(lbl_path).convert('RGB')
        except Exception as e:
            print(f"加载样本 {idx} 失败: {str(e)}，使用默认样本")
            image = Image.new('RGB', self.fixed_size, (0, 0, 0))
            label = Image.new('RGB', self.fixed_size, (0, 0, 0))

        # 强制缩放到固定尺寸（核心：确保所有样本尺寸统一）
        image = F.resize(image, self.fixed_size, interpolation=T.InterpolationMode.BILINEAR)
        label = F.resize(label, self.fixed_size, interpolation=T.InterpolationMode.NEAREST)

        # 数据增强（仅训练时使用）
        if self.augment:
            image, label = self._apply_augmentations(image, label)
            # 增强后再次确保尺寸正确
            image = F.resize(image, self.fixed_size, interpolation=T.InterpolationMode.BILINEAR)
            label = F.resize(label, self.fixed_size, interpolation=T.InterpolationMode.NEAREST)

        image = F.to_tensor(image)  # 此时是(3, 736, 1280)，如果变成了(3,1280,736)，添加转置
        label = F.to_tensor(label)  # 同理

        # 关键修复：强制转置，确保维度是 (C,736,1280)
        if image.shape == (3, 1280, 736):
            image = image.permute(0, 2, 1)  # (3,1280,736) → (3,736,1280)
        if label.shape == (3, 1280, 736):
            label = label.permute(0, 2, 1)

        # 标签二值化
        label = (label.sum(dim=0, keepdim=True) > 0).float()

        # 再次校验标签维度
        if label.shape == (1, 1280, 736):
            label = label.permute(0, 2, 1)  # (1,1280,736) → (1,736,1280)

        # 应用外部transform（归一化）
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _apply_augmentations(self, image, label):
        """增强版数据增强，防止过拟合"""
        # 随机水平翻转
        if random.random() < 0.5:
            image = F.hflip(image)
            label = F.hflip(label)

        # 随机垂直翻转
        if random.random() < 0.2:
            image = F.vflip(image)
            label = F.vflip(label)

        # 随机旋转（±8度）
        if random.random() < 0.4:
            angle = random.uniform(-8, 8)
            image = F.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
            label = F.rotate(label, angle, interpolation=T.InterpolationMode.NEAREST)

        # 随机缩放（0.8-1.2倍）
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            w, h = self.fixed_size
            new_w, new_h = int(w * scale), int(h * scale)
            # 先缩放
            image = F.resize(image, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)
            label = F.resize(label, (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)
            # 再中心裁剪回固定尺寸
            image = F.center_crop(image, (h, w))
            label = F.center_crop(label, (h, w))

        # 随机仿射变换
        if random.random() < 0.3:
            shear_factor = random.uniform(-5, 5)
            scale_factor = random.uniform(0.9, 1.1)
            image = F.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=scale_factor,
                shear=shear_factor,
                interpolation=T.InterpolationMode.BILINEAR
            )
            label = F.affine(
                label,
                angle=0,
                translate=[0, 0],
                scale=scale_factor,
                shear=shear_factor,
                interpolation=T.InterpolationMode.NEAREST
            )

        # 随机亮度/对比度/饱和度调整
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.4, 1.6)
            image = F.adjust_brightness(image, brightness_factor)
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.4, 1.6)
            image = F.adjust_contrast(image, contrast_factor)
        if random.random() < 0.5:
            saturation_factor = random.uniform(0.4, 1.6)
            image = F.adjust_saturation(image, saturation_factor)

        # 随机伽马校正
        if random.random() < 0.3:
            gamma = random.uniform(0.6, 1.4)
            image = F.adjust_gamma(image, gamma)

        # 随机裁剪
        if random.random() < 0.4:
            w, h = self.fixed_size
            crop_scale = random.uniform(0.8, 1.0)
            new_w = int(w * crop_scale)
            new_h = int(h * crop_scale)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            # 裁剪后缩放回固定尺寸
            image_crop = F.crop(image, top, left, new_h, new_w)
            label_crop = F.crop(label, top, left, new_h, new_w)
            image = F.resize(image_crop, (h, w), interpolation=T.InterpolationMode.BILINEAR)
            label = F.resize(label_crop, (h, w), interpolation=T.InterpolationMode.NEAREST)

        # 随机高斯噪声
        if random.random() < 0.3:
            img_np = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, 0.03, img_np.shape)
            img_np = np.clip(img_np + noise, 0, 1)
            image = Image.fromarray((img_np * 255).astype(np.uint8))

        # 随机高斯模糊
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            image = T.GaussianBlur(ksize)(image)

        return image, label

    def _load_annotations(self):
        """加载标注文件"""
        ann_dir = self.root
        json_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]

        for json_name in json_files:
            json_path = os.path.join(ann_dir, json_name)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        ann = json.loads(line)

                        # 构建图像和标签路径
                        img_rel = ann["raw_file"].replace("/", os.sep)
                        img_rel = os.path.splitext(img_rel)[0] + ".jpg"

                        lbl_rel = img_rel.replace("clips", "seg_label")
                        lbl_rel = os.path.splitext(lbl_rel)[0] + ".png"

                        # 验证文件是否存在
                        full_img_path = os.path.join(self.root, img_rel)
                        full_lbl_path = os.path.join(self.root, lbl_rel)

                        if os.path.exists(full_img_path) and os.path.exists(full_lbl_path):
                            self.images.append(img_rel)
                            self.labels.append(lbl_rel)

            except Exception as e:
                print(f"加载标注文件 {json_name} 出错: {str(e)}")
                continue


if __name__ == "__main__":
    # 测试数据集
    transform = T.Compose([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = LaneDetectionDataset(
        root=r"/root/autodl-tmp/kaggle_cache/datasets/manideep1108/tusimple/versions/5/TUSimple/train_set",
        transform=transform,
        split="train_set",
        augment=True
    )
    print(f"数据集大小: {len(ds)}")

    # 测试数据加载
    img, lbl = ds[0]
    print(f"image shape: {img.shape}, label shape: {lbl.shape}")
    print(f"label min: {lbl.min()}, max: {lbl.max()}")
    lane_pixel_ratio = (lbl == 1).sum() / lbl.numel()
    print(f"车道线像素占比: {lane_pixel_ratio:.4f}")
