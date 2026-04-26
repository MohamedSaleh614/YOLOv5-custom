import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSILU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2 if stride == 1 else (kernel_size - 1) // 2
            if kernel_size == 6:
                padding = 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.bn(self.conv2d(x)))

class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.conv1 = ConvBNSILU(channels, channels, kernel_size=1)
        self.conv2 = ConvBNSILU(channels, channels, kernel_size=3)
        self.use_shortcut = shortcut

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x + x2 if self.use_shortcut else x2

class C3(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks=3, shortcut=True):
        super().__init__()
        self.conv1 = ConvBNSILU(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBNSILU(in_channels, out_channels, kernel_size=1)
        self.conv3 = ConvBNSILU(2 * out_channels, out_channels, kernel_size=1)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(out_channels, shortcut=shortcut) for _ in range(num_bottlenecks)]
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.bottlenecks(y2)
        y = torch.cat((y2, y1), dim=1)
        return self.conv3(y)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNSILU(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBNSILU(out_channels * 4, out_channels, kernel_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)
        x3 = self.maxpool2(x2)
        x4 = self.maxpool3(x3)
        x5 = torch.cat((x1, x2, x3, x4), dim=1)
        return self.conv2(x5)

class YOLOv5(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, width_multiple=0.50, depth_multiple=0.67):
        super().__init__()
        def make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        def ch(v):
            return make_divisible(v * width_multiple)
        def dep(v):
            return max(round(v * depth_multiple), 1) if v > 1 else v

        self.conv1 = ConvBNSILU(in_channels, ch(64), kernel_size=6, stride=2)
        self.conv2 = ConvBNSILU(ch(64), ch(128), kernel_size=3, stride=2)
        self.c3_1 = C3(ch(128), ch(128), num_bottlenecks=dep(3), shortcut=True)
        self.conv3 = ConvBNSILU(ch(128), ch(256), kernel_size=3, stride=2)
        self.c3_2 = C3(ch(256), ch(256), num_bottlenecks=dep(6), shortcut=True)
        self.conv4 = ConvBNSILU(ch(256), ch(512), kernel_size=3, stride=2)
        self.c3_3 = C3(ch(512), ch(512), num_bottlenecks=dep(9), shortcut=True)
        self.conv5 = ConvBNSILU(ch(512), ch(1024), kernel_size=3, stride=2)
        self.c3_4 = C3(ch(1024), ch(1024), num_bottlenecks=dep(3), shortcut=True)
        self.sppf = SPPF(ch(1024), ch(1024))

        self.conv6 = ConvBNSILU(ch(1024), ch(512), kernel_size=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_5 = C3(ch(1024), ch(512), num_bottlenecks=dep(3), shortcut=False)
        self.conv7 = ConvBNSILU(ch(512), ch(256), kernel_size=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_6 = C3(ch(512), ch(256), num_bottlenecks=dep(3), shortcut=False)
        self.conv8 = ConvBNSILU(ch(256), ch(256), kernel_size=3, stride=2)
        self.c3_7 = C3(ch(512), ch(512), num_bottlenecks=dep(3), shortcut=False)
        self.conv9 = ConvBNSILU(ch(512), ch(512), kernel_size=3, stride=2)
        self.c3_8 = C3(ch(1024), ch(1024), num_bottlenecks=dep(3), shortcut=False)

        self.conv_small  = nn.Conv2d(ch(256), 3 * (5 + num_classes), kernel_size=1)
        self.conv_medium = nn.Conv2d(ch(512), 3 * (5 + num_classes), kernel_size=1)
        self.conv_large  = nn.Conv2d(ch(1024), 3 * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.c3_1(x2)
        x4 = self.conv3(x3)
        x5 = self.c3_2(x4)
        x6 = self.conv4(x5)
        x7 = self.c3_3(x6)
        x8 = self.conv5(x7)
        x9 = self.c3_4(x8)
        x10 = self.sppf(x9)

        x11 = self.conv6(x10)
        x12 = self.upsample1(x11)
        if x12.shape[2:] != x7.shape[2:]:
            x12 = F.pad(x12, [0, x7.shape[3]-x12.shape[3], 0, x7.shape[2]-x12.shape[2]])
        x13 = torch.cat((x12, x7), dim=1)
        x14 = self.c3_5(x13)

        x15 = self.conv7(x14)
        x16 = self.upsample2(x15)
        if x16.shape[2:] != x5.shape[2:]:
            x16 = F.pad(x16, [0, x5.shape[3]-x16.shape[3], 0, x5.shape[2]-x16.shape[2]])
        x17 = torch.cat((x16, x5), dim=1)
        x18 = self.c3_6(x17)

        x19 = self.conv8(x18)
        x20 = torch.cat((x19, x15), dim=1)
        x21 = self.c3_7(x20)

        x22 = self.conv9(x21)
        x23 = torch.cat((x22, x11), dim=1)
        x24 = self.c3_8(x23)

        out_small  = self.conv_small(x18)
        out_medium = self.conv_medium(x21)
        out_large  = self.conv_large(x24)

        return [out_small, out_medium, out_large]