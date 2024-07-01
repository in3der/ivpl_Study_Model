import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self, in_channels=None, out_channels=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Conv1x3_3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x3_3x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1), stride=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0), stride=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv2(self.relu(self.bn(self.conv1(self.relu(x))))))

class Conv1x7_7x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x7_7x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 7), padding=(0, 3), stride=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0), stride=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv2(self.relu(self.bn(self.conv1(self.relu(x))))))

class AvgPool3x3(nn.Module):
    def __init__(self):
        super(AvgPool3x3, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, 3, stride=1, padding=1)

class MaxPool3x3(nn.Module):
    def __init__(self):
        super(MaxPool3x3, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 3, stride=1, padding=1)

class MaxPool5x5(nn.Module):
    def __init__(self):
        super(MaxPool5x5, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 5, stride=1, padding=2)

class MaxPool7x7(nn.Module):
    def __init__(self):
        super(MaxPool7x7, self).__init__()

    def forward(self, x):
        return F.max_pool2d(x, 7, stride=1, padding=3)


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(self.relu(x)))

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(self.relu(x)))


class DilatedConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(self.relu(x)))

class DepthwiseSeparableConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv3x3, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.pointwise(self.depthwise(self.relu(x)))



class DepthwiseSeparableConv5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv5x5, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=5 // 2, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.pointwise(self.depthwise(self.relu(x)))


class DepthwiseSeparableConv7x7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv7x7, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=7 // 2, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(self.relu(x))
        return x


operations = {
    'identity': Identity,
    '1x3 then 3x1 convolution': Conv1x3_3x1,
    '1x7 then 7x1 convolution': Conv1x7_7x1,
    '3x3 average pooling': AvgPool3x3,
    '3x3 max pooling': MaxPool3x3,
    '5x5 max pooling': MaxPool5x5,
    '7x7 max pooling': MaxPool7x7,
    '1x1 convolution': Conv1x1,
    '3x3 convolution': Conv3x3,
    '3x3 dilated convolution': DilatedConv3x3,
    '3x3 depthwise-separable conv': DepthwiseSeparableConv3x3,
    '5x5 depthwise-separable conv': DepthwiseSeparableConv5x5,
    '7x7 depthwise-separable conv': DepthwiseSeparableConv7x7,
}
