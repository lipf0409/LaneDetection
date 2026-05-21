# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models


# ---------------- 第一步：完全复刻服务器训练的模型结构（关键！） ----------------
class LaneDetectionModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # 服务器训练时的backbone：直接使用ResNet的原始结构（不封装Sequential）
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 冻结前两层（和服务器完全一致）
        for param in list(resnet.conv1.parameters()) + list(resnet.bn1.parameters()):
            param.requires_grad = False

        # 直接赋值ResNet的层（保留原始命名：conv1/bn1/layer1/layer2/layer3）
        self.backbone_conv1 = resnet.conv1
        self.backbone_bn1 = resnet.bn1
        self.backbone_relu = resnet.relu
        self.backbone_maxpool = resnet.maxpool
        self.backbone_layer1 = resnet.layer1
        self.backbone_layer2 = resnet.layer2
        self.backbone_layer3 = resnet.layer3

        # 上采样模块（维度和服务器完全一致）
        self.up4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.classifier = nn.Conv2d(16, 1, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """和服务器训练代码一致的初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 完全复刻服务器的forward逻辑（保留ResNet原始层调用）
        x = self.backbone_conv1(x)
        x = self.backbone_bn1(x)
        x = self.backbone_relu(x)
        x = self.backbone_maxpool(x)

        x = self.backbone_layer1(x)
        x = self.backbone_layer2(x)
        x = self.backbone_layer3(x)

        # 上采样流程
        out = self.up4(x)
        out = torch.relu(self.bn4(self.conv4(out)))

        out = self.up3(out)
        out = torch.relu(self.bn3(self.conv3(out)))

        out = self.up2(out)
        out = torch.relu(self.bn2(self.conv2(out)))

        out = self.up1(out)
        return self.classifier(out)

    def rename_weights(self, state_dict):
        """关键：将服务器权重的键名适配到本地模型"""
        new_state_dict = {}
        for k, v in state_dict.items():
            # 处理backbone的键名映射（服务器→本地）
            if k.startswith('backbone.conv1'):
                new_k = k.replace('backbone.conv1', 'backbone_conv1')
            elif k.startswith('backbone.bn1'):
                new_k = k.replace('backbone.bn1', 'backbone_bn1')
            elif k.startswith('backbone.layer1'):
                new_k = k.replace('backbone.layer1', 'backbone_layer1')
            elif k.startswith('backbone.layer2'):
                new_k = k.replace('backbone.layer2', 'backbone_layer2')
            elif k.startswith('backbone.layer3'):
                new_k = k.replace('backbone.layer3', 'backbone_layer3')
            else:
                # 其他层（up4/conv4等）键名不变
                new_k = k
            new_state_dict[new_k] = v
        return new_state_dict


# ---------------- 配置参数 ----------------
CFG = {
    'model_path': r'E:\PY_CODE\LaneDetection-main\best-ckpt1.pt',
    'input_size': (736, 1280),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'threshold': 0.01,
    'input_dir': r"E:\PY_CODE\LaneDetection-main",
    'output_dir': r'E:\PY_CODE\LaneDetection-main\results'
}


# ---------------- 工具函数 ----------------
def preprocess_image(img_path, input_size=(736, 1280)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((input_size[1], input_size[0]), Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, np.array(img)


def visualize_result(raw_img, pred_mask, threshold=0.5):
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
    mask_color = np.zeros_like(raw_img)
    mask_color[:, :, 2] = pred_mask
    result = cv2.addWeighted(raw_img, 0.7, mask_color, 0.3, 0)
    return result


# ---------------- 预测函数 ----------------
def predict_single_image(model, img_path):
    img_tensor, raw_img = preprocess_image(img_path, CFG['input_size'])
    img_tensor = img_tensor.to(CFG['device'])

    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = pred.squeeze().cpu().numpy()

    result_img = visualize_result(raw_img, pred_mask, CFG['threshold'])
    return raw_img, pred_mask, result_img


def batch_predict():
    os.makedirs(CFG['output_dir'], exist_ok=True)

    # 加载模型（终极修复：适配权重命名）
    print(f"正在加载模型：{CFG['model_path']}")
    model = LaneDetectionModel(pretrained=False)

    # 加载服务器权重
    checkpoint = torch.load(
        CFG['model_path'],
        map_location=CFG['device'],
        weights_only=True
    )
    # 提取state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # 关键步骤：重命名权重键名，匹配本地模型
    new_state_dict = model.rename_weights(state_dict)

    # 加载权重（忽略无关的键，确保核心权重加载）
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(CFG['device'])
    print(f"模型加载完成！使用设备：{CFG['device']}")

    # 获取测试图片
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = []
    for f in os.listdir(CFG['input_dir']):
        if os.path.splitext(f)[1].lower() in img_extensions:
            img_paths.append(os.path.join(CFG['input_dir'], f))

    if not img_paths:
        print(f"错误：未找到图片！路径：{CFG['input_dir']}")
        return

    # 批量预测
    print(f"找到 {len(img_paths)} 张图片，开始预测...")
    for idx, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        print(f"处理：{img_name} ({idx + 1}/{len(img_paths)})")

        try:
            raw_img, pred_mask, result_img = predict_single_image(model, img_path)

            # 保存结果
            result_path = os.path.join(CFG['output_dir'], f"result_{img_name}")
            cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            mask_path = os.path.join(CFG['output_dir'], f"mask_{img_name}")
            cv2.imwrite(mask_path, (pred_mask > CFG['threshold']).astype(np.uint8) * 255)

        except Exception as e:
            print(f"处理 {img_name} 失败：{str(e)}")
            continue

    print(f"\n预测完成！结果保存至：{CFG['output_dir']}")


# ---------------- 运行 ----------------
if __name__ == '__main__':
    batch_predict()
