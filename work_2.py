# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn

class Combine_1(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Combine_1,self).__init__(*args, **kwargs)
        self.W1 = nn.Parameter(torch.rand(12,90,90))
        self.W2 = nn.Parameter(torch.rand(12,90,90))
        self.W3 = nn.Parameter(torch.rand(12,90,90))

    def forward(self, x1, x2, x3):
        h_1 = torch.mul(self.W1, x1)
        h_2 = torch.mul(self.W2, x2)
        h_3 = torch.mul(self.W3, x3)
        return h_1 + h_2 + h_3

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
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, size):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net_Com(nn.Module):

    def __init__(self, in_ch=12, out_ch=12):
        super(U_Net_Com, self).__init__()

        # 卷积参数设置
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4]

        # 最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 左边特征提取卷积层,分别针对x_1, x_2, x_3
        self.Conv1_1 = conv_block(in_ch, filters[0])
        self.Conv2_1 = conv_block(filters[0], filters[1])
        self.Conv3_1 = conv_block(filters[1], filters[2])

        self.Conv1_2 = conv_block(in_ch, filters[0])
        self.Conv2_2 = conv_block(filters[0], filters[1])
        self.Conv3_2 = conv_block(filters[1], filters[2])

        self.Conv1_3 = conv_block(in_ch, filters[0])
        self.Conv2_3 = conv_block(filters[0], filters[1])
        self.Conv3_3 = conv_block(filters[1], filters[2])

        # 右边特征融合反卷积层, 一层一层对高低语义信息进行拼接

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

        d3 = self.Up3(e3_1+e3_2+e3_3)
        e2 = e2_1 + e2_2 + e2_3
        d3 = torch.cat((e2, d3), dim=1)  # 将e2特征图与d3特征图横向拼接
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1 = e1_1 + e1_2 + e1_3
        d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out.squeeze()

class PCCloss(nn.Module):
    def __init__(self):
        super(PCCloss,self).__init__()
        self.batch_size = 12

    def forward(self, y_pred, y_true):

        PRED = y_pred.reshape(-1,1)
        TRUE = y_true.reshape(-1,1)
        COV = torch.cov(torch.cat([PRED,TRUE], dim=1))[0][1]
        std_p = torch.std(PRED)
        std_t = torch.std(TRUE)
        VAR = std_p * std_t
        out = torch.div(COV, VAR)
        return out


if __name__ == "__main__":
    epochs = 300
    patience = 25
    X_1 = torch.rand(12,90,90)
    X_2 = torch.rand(12,90,90)
    X_3 = torch.rand(12,90,90)
    Label = torch.rand(12,90,90)
    model = U_Net_Com()
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    PCC = PCCloss()
    loss = nn.MSELoss()
    best_loss = 1e9
    cnt = 0
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_1, X_2, X_3)
        batch_loss = loss(pred, Label)
        pcc_loss = PCC(pred, Label)
        total_loss = batch_loss+pcc_loss
        total_loss.backward()
        optimizer.step()
        print(f"epochs: {i}, loss: {total_loss.item()}")
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            cnt = 0
        else:
            cnt += 1
        if cnt == patience:
            break

