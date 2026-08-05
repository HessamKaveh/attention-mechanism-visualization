import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) block: وزن‌دهی کانال‌های ویژگی"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        pool = self.global_pool(x).view(B, C)
        weights = self.fc(pool).view(B, C, 1, 1)
        return x * weights


class SpatialAttention(nn.Module):
    """Spatial Attention: وزن‌دهی موقعیت‌های مکانی در ویژگی‌ها"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ch_avg = x.mean(dim=1, keepdim=True)
        ch_max = x.max(dim=1, keepdim=True)[0]
        cat = torch.cat([ch_avg, ch_max], dim=1)
        weights = self.conv(cat)
        return x * weights


class AttentionBlock(nn.Module):
    """ترکیب Channel + Spatial Attention"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.att1 = AttentionBlock(32)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.att2 = AttentionBlock(64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.att3 = AttentionBlock(128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.att1(x)

        x = self.conv2(x)
        x = self.att2(x)

        x = self.conv3(x)
        x = self.att3(x)

        x = self.pool(x).flatten(1)
        return self.fc(x)
