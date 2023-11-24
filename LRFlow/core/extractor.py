import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, inc, outc, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(True)

        self.norm1 = nn.InstanceNorm2d(outc)
        self.norm2 = nn.InstanceNorm2d(outc)
        self.norm3 = nn.InstanceNorm2d(outc)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
                self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample:
            x = self.downsample(x)

        return self.relu(x+y)

class BlockEncoder(nn.Module):
    def __init__(self, outc=256, dropout=0.0):
        super(BlockEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.norm1 = nn.BatchNorm2d(64)

        if dropout ==0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128, outc, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResBlock(self.in_planes, dim, stride=stride)
        layer2 = ResBlock(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
