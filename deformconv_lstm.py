import torch
import torch.nn as nn
import torch.nn.functional as F
from deform_cov import DeformConv2D
from SE_block import *


class DeformLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(DeformLSTM, self).__init__()
        self.hidden_dim = 10
        self.num_layers = 2
        self.offset1 = nn.Conv2d(in_channel, 18, kernel_size=3, padding=1)
        self.conv1 = DeformConv2D(in_channel, 64, kernel_size=3, padding=1)
        self.embed1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.offset2 = nn.Conv2d(64, 18, kernel_size=3, padding=1)
        self.conv2 = DeformConv2D(64, 64, kernel_size=3, padding=1)
        self.embed2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.offset3 = nn.Conv2d(64, 18, kernel_size=3, padding=1)
        self.conv3 = DeformConv2D(64, 128, kernel_size=3, padding=1)
        self.embed3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.offset4 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.embed4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.offset5 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv5 = DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.embed5 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.offset6 = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv6 = DeformConv2D(128, 256, kernel_size=3, padding=1)
        self.embed6 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.offset7 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.conv7 = DeformConv2D(256, 256, kernel_size=3, padding=1)
        self.embed7 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.offset8 = nn.Conv2d(256, 18, kernel_size=3, padding=1)
        self.conv8 = DeformConv2D(256, 256, kernel_size=3, padding=1)
        self.embed8 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.bilstm = nn.LSTM(9, self.hidden_dim,
                              num_layers=self.num_layers, bidirectional=False,
                              batch_first=True, bias=False)

        self.hidden2label1 = nn.Sequential(nn.Linear(256 * self.hidden_dim, 400), nn.ReLU(),
                                           nn.Dropout())
        self.hidden2label2 = nn.Linear(400, out_channel)

    def forward(self, x):

        offset1 = self.offset1(x)
        x = self.conv1(x, offset1)
        x = self.embed1(x)

        offset2 = self.offset2(x)
        x = self.conv2(x, offset2)
        x = self.embed2(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        offset3 = self.offset3(x)
        x = self.conv3(x, offset3)
        x = self.embed3(x)

        offset4 = self.offset4(x)
        x = self.conv4(x, offset4)
        x = self.embed4(x)

        offset5 = self.offset5(x)
        x = self.conv5(x, offset5)
        x = self.embed5(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        offset6 = self.offset6(x)
        x = self.conv6(x, offset6)
        x = self.embed6(x)

        offset7 = self.offset7(x)
        x = self.conv7(x, offset7)
        x = self.embed7(x)

        offset8 = self.offset8(x)
        x = self.conv8(x, offset8)
        x = self.embed8(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 256, 9)

        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), -1)
        logit = self.hidden2label1(bilstm_out)
        logit = self.hidden2label2(logit)

        return logit


# net = DeformLSTM()

# from torchkeras import summary
# import torch
#
# print(summary(net, input_shape=(1,24,24)))