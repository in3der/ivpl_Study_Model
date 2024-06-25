import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def forward(self, x):
        return x

class Conv1x3_3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x3_3x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))

    def forward(self, x):
        return self.conv2(self.conv1(x))

class Conv1x7_7x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x7_7x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 7), padding=(0, 3))
        self.conv2 = nn.Conv2d(out_channels, out_channels, (7, 1), padding=(3, 0))

    def forward(self, x):
        return self.conv2(self.conv1(x))

class AvgPool3x3(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 3, stride=1, padding=1)

class MaxPool3x3(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, 3, stride=1, padding=1)

class MaxPool5x5(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, 5, stride=1, padding=2)

class MaxPool7x7(nn.Module):
    def forward(self, x):
        return F.max_pool2d(x, 7, stride=1, padding=3)

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)

class Conv1x3_3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x3_3x1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 1), padding=(1, 0))

    def forward(self, x):
        return self.conv2(self.conv1(x))

class DilatedConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv3x3, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv5x5, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 5, groups=in_channels, padding=2)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 12)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class DepthwiseSeparableConv7x7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv7x7, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 7, groups=in_channels, padding=3)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


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
