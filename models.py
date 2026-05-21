import torch
import torch.nn as nn
import torchvision.models as models


class SELayer(nn.Module):
    """ squeeze-excitation注意力模块 """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.1):  # 新增dropout参数
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # 冻结前两层，只训练后面的层
        for param in list(resnet.conv1.parameters()) + list(resnet.bn1.parameters()):
            param.requires_grad = False

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 ch
        self.layer2 = resnet.layer2  # 128 ch
        self.layer3 = resnet.layer3  # 256 ch

        # 添加注意力模块
        self.se1 = SELayer(64)
        self.se2 = SELayer(128)
        self.se3 = SELayer(256)

        # 添加dropout层减少过拟合
        self.dropout = nn.Dropout2d(dropout_rate)  # 空间dropout

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f4 = self.layer1(x)  # 1/4  64 ch
        f4 = self.se1(f4)
        f4 = self.dropout(f4)  # 应用dropout

        f8 = self.layer2(f4)  # 1/8  128 ch
        f8 = self.se2(f8)
        f8 = self.dropout(f8)  # 应用dropout

        f16 = self.layer3(f8)  # 1/16 256 ch
        f16 = self.se3(f16)
        f16 = self.dropout(f16)  # 应用dropout

        return f4, f8, f16


class LaneDetectionModel(nn.Module):
    def __init__(self, pretrained=False, dropout_rate=0.1):  # 新增dropout参数
        super().__init__()
        self.backbone = ResNetBackbone(pretrained, dropout_rate)  # 传递dropout参数

        # 上采样模块：添加跳跃连接 + 改进卷积
        # 1/16 → 1/8
        self.up4 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),  # 融合f8特征
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # 新增dropout
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128)
        )

        # 1/8 → 1/4
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),  # 融合f4特征
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),  # 新增dropout
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )

        # 1/4 → 1/2
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)  # 新增dropout
        )

        # 1/2 → 1/1
        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.classifier = nn.Conv2d(16, 1, 1)

        # 初始化转置卷积权重
        self._init_weights()

    def _init_weights(self):
        """初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f4, f8, f16 = self.backbone(x)  # 1/4 1/8 1/16

        # 上采样并融合跳跃连接特征
        out = self.up4(f16)
        out = torch.cat([out, f8], dim=1)  # 融合1/8特征
        out = torch.relu(self.conv4(out))

        out = self.up3(out)
        out = torch.cat([out, f4], dim=1)  # 融合1/4特征
        out = torch.relu(self.conv3(out))

        out = self.up2(out)
        out = torch.relu(self.conv2(out))

        out = self.up1(out)
        return self.classifier(out)
