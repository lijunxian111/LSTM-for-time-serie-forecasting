import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

weight = Parameter(torch.FloatTensor(1.0, 1.0, 1.0))

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(up_conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(size=size, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net_Com(nn.Module):
    def __init__(self, in_ch=12, out_ch=12):
        super(U_Net_Com, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(in_ch, filters[0])
        self.Conv2_1 = conv_block(filters[0], filters[1])
        self.Conv3_1 = conv_block(filters[1], filters[2])

        self.Conv1_2 = conv_block(in_ch, filters[0])
        self.Conv2_2 = conv_block(filters[0], filters[1])
        self.Conv3_2 = conv_block(filters[1], filters[2])

        self.Conv1_3 = conv_block(in_ch, filters[0])
        self.Conv2_3 = conv_block(filters[0], filters[1])
        self.Conv3_3 = conv_block(filters[1], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], 45)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], 90)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3):
        x1 = torch.reshape(x1, shape=[1, 12, 90, 90])
        x2 = torch.reshape(x2, shape=[1, 12, 90, 90])
        x3 = torch.reshape(x3, shape=[1, 12, 90, 90])
        e1_1 = self.Conv1_1(x1)
        e2_1 = self.Maxpool1(e1_1)
        e2_1 = self.Conv2_1(e2_1)
        e3_1 = self.Maxpool2(e2_1)
        e3_1 = self.Conv3_1(e3_1)

        e1_2 = self.Conv1_2(x2)
        e2_2 = self.Maxpool1(e1_2)
        e2_2 = self.Conv2_2(e2_2)
        e3_2 = self.Maxpool2(e2_2)
        e3_2 = self.Conv3_2(e3_2)

        e1_3 = self.Conv1_3(x3)
        e2_3 = self.Maxpool1(e1_3)
        e2_3 = self.Conv2_3(e2_3)
        e3_3 = self.Maxpool2(e2_3)
        e3_3 = self.Conv3_3(e3_3)

        d3 = self.Up3(e3_1 + e3_2 + e3_3)
        e2 = e2_1 + e2_2 + e2_3
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1 = e1_1 + e1_2 + e1_3
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out.squeeze()

x1_1 = torch.rand(12, 90, 90)
x1_2 = torch.rand(12, 90, 90)
x1_3 = torch.rand(12, 90, 90)

x2_1 = torch.rand(12, 90, 90)
x2_2 = torch.rand(12, 90, 90)
x2_3 = torch.rand(12, 90, 90)

x3_1 = torch.rand(12, 90, 90)
x3_2 = torch.rand(12, 90, 90)
x3_3 = torch.rand(12, 90, 90)

model1 = U_Net_Com()
out1 = model1(x1_1, x1_2, x1_3)

model2 = U_Net_Com()
out2 = model2(x2_1, x2_2, x2_3)

model3 = U_Net_Com()
out3 = model3(x3_1, x3_2, x3_3)

out = weight[0] * out1 + weight[1] * out2 + weight[2] * out3












