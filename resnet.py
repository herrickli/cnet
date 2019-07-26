from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, channels, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(channels),
            nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(channels),
            nn.Conv2d(channels, self.expansion * channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.expansion * channels)
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.conv(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)


class Resnet101(nn.Module):
    def __init__(self, num_class=5):
        super(Resnet101, self).__init__()
        self.in_channel = 64
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 23, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_class)

    def _make_layer(self, channels, num_block, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(self.in_channel, channels * 4, 1, stride, bias=False),
            nn.BatchNorm2d(channels * 4)
        )
        layers = []
        layers.append(Bottleneck(self.in_channel, channels, stride, shortcut))
        self.in_channel = channels * 4
        for i in range(num_block):
            layers.append(Bottleneck(self.in_channel, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        return x
        #x = x.view(x.size(0), -1)
        #return self.fc(x)

if __name__ == '__main__':
    model = Resnet101()
    import torch
    input = torch.randn(1, 3, 1000, 600)
    o = model(input)
    print(o.shape)
