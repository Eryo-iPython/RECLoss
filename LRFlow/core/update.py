import torch
from torch import nn

class DepthWiseConv(nn.Module):
    def __init__(self, inc, outc, kernel_size, padding=0, stride=1):
        super(DepthWiseConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=kernel_size, padding=padding,
                      stride=stride, groups=inc),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inc, outc, 1),
                    )

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        return out

class FlowEncoder(nn.Module):
    def __init__(self, args):
        super(FlowEncoder, self).__init__()
        self.convf = nn.Sequential(
            nn.Conv2d(2, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        dim = (2*args.radius + 1)**2
        self.corr = nn.Sequential(
            nn.Conv2d(dim, 256, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 192, 3, padding=1),
            nn.ReLU()
        )
        self.sum = nn.Sequential(
            nn.Conv2d(192+64, 128-2, 3, padding=1),
            nn.ReLU()
        )


    def forward(self, flow, correlation):

        corr = self.corr(correlation)
        flow_ = self.convf(flow)

        x = torch.cat([corr, flow_], dim=1)
        x = self.sum(x)
        return torch.cat([x, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = DepthWiseConv(hidden_dim+input_dim, hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.convr1 = DepthWiseConv(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = DepthWiseConv(hidden_dim+input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = DepthWiseConv(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = DepthWiseConv(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = DepthWiseConv(hidden_dim+input_dim, hidden_dim, (5, 1), padding=(2, 0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class ToFlow(nn.Module):
    def __init__(self, inc=64, h_dim=256):
        super(ToFlow, self).__init__()
        self.conv1 = nn.Conv2d(inc, h_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(h_dim, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class UpdateBlock(nn.Module):
    def __init__(self, h_dim=128, args=None):
        super(UpdateBlock, self).__init__()
        self.encoder = FlowEncoder(args=args)
        self.gru = SepConvGRU(h_dim, input_dim=h_dim+128)
        self.toflow = ToFlow(inc=h_dim, h_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(h_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
        )

    def forward(self, f1, f2, flow):
        # 128
        flow_features = self.encoder(flow, f2)

        #[b, 128+128, h, w]
        inp = torch.cat([f1, flow_features], dim=1)

        net = self.gru(f1, inp)
        delta_flow = self.toflow(net)

        mask = .25 * self.mask(net)
        return net, mask, delta_flow