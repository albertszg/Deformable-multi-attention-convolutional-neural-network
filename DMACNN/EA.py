import torch.nn as nn
import torch.nn.functional as F
from deform_cov import DeformConv2D
from SE_block import *
import torch
#单独空间注意力机制
class EAconvLSTM(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(EAconvLSTM, self).__init__()
        self.hidden_dim = 10
        self.num_layers = 2

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.embed1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.ea1=EALayer(64,16)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.embed2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.ea2= EALayer(64, 16)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.embed3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.ea3= EALayer(128, 16)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.embed4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.ea4= EALayer(128, 16)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.embed5 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.ea5= EALayer(128, 16)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.embed6 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.ea6= EALayer(256, 16)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.embed7 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.ea7= EALayer(256, 16)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.embed8 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.ea8= EALayer(256, 16)

        self.bilstm = nn.LSTM(9, self.hidden_dim,
                              num_layers=self.num_layers, bidirectional=False,
                              batch_first=True, bias=False)

        self.hidden2label1 = nn.Sequential(nn.Linear(256 * self.hidden_dim, 400), nn.ReLU(),
                                           nn.Dropout())
        self.hidden2label2 = nn.Linear(400, out_channel)

    def forward(self, x):

        x = self.conv1(x)
        x = self.embed1(x)
        x= self.ea1(x)

        x = self.conv2(x)
        x = self.embed2(x)
        x= self.ea2(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.embed3(x)
        x= self.ea3(x)

        x = self.conv4(x)
        x = self.embed4(x)
        x = self.ea4(x)

        x = self.conv5(x)
        x = self.embed5(x)

        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.ea5(x)

        x = self.conv6(x)
        x = self.embed6(x)
        x = self.ea6(x)

        x = self.conv7(x)
        x = self.embed7(x)
        x = self.ea7(x)

        x = self.conv8(x)
        x = self.embed8(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.ea8(x)

        x = x.view(-1, 256, 9)

        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), -1)
        logit = self.hidden2label1(bilstm_out)
        logit = self.hidden2label2(logit)

        return logit


# net = convLSTM()
#
# from torchkeras import summary
# import torch
#
# print(summary(net, input_shape=(1,24,24)))