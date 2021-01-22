from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

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

        return x * y.expand_as(x)+x


class EALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(EALayer, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel,out_channels=1, kernel_size=1,bias=False),
            nn.Sigmoid()#  0~1 attention weights
        )
        #卷积完要保证跟原输入一致，通道等完全一致，
        self.conv2=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=3,padding=1)
        # self.conv2 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.conv1(x).view(b, 1, h, w)
        # 防止对脉冲响应信号的过聚焦
        x=self.conv2(x)
        x=self.relu(x)
        EA_W=y.expand_as(x)
        return x * EA_W+x