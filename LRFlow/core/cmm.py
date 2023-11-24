from torch import nn
from torch.nn import functional as F
from utils.utils import bilinear_sampler
import torch


class CorrelationMatchingModule(nn.Module):
    def __init__(self, radius=1):
        super(CorrelationMatchingModule, self).__init__()
        self.r = radius

    #
    def warp(self, img, vgrid):
        vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
        img = bilinear_sampler(img, vgrid)
        return img


    def corr_softmax(self, q, k, v, c0, final=False):
        #q k v  radius: f1, f2, c1, iter_nums
        b, c, h, w = q.shape
        scale_factor = c ** 0.5
        init_grid = c0

        #learnable offset
        dx = torch.linspace(-self.r, self.r, 2 * self.r + 1)
        dy = torch.linspace(-self.r, self.r, 2 * self.r + 1)
        delta_grid = torch.meshgrid(dy, dx)
        delta = torch.stack(delta_grid[::-1], axis=-1).float().to(q.device)
        delta = delta.reshape(-1, 2).repeat(b, 1, 1, 1)
        algha = torch.autograd.Variable(torch.ones_like(delta)) # B 1 (2R+1)^2 2
        delta = algha * delta

        init_grid = init_grid.reshape(b, 2, h*w).permute(0, 2, 1).contiguous()
        init_grid = init_grid.unsqueeze(-2)

        x = init_grid + delta # B, H*W, (2R+1)^2, 2


        #warped f2
        window_feature = bilinear_sampler(k, x)
        window_feature = window_feature.permute(0, 2, 1, 3) #[B, H*W, C, (2R+1)^2]

        q = q.reshape(b, c, h*w).permute(0, 2, 1).contiguous().unsqueeze(-2)

        correlation = (q@window_feature).reshape(b, h*w, -1) / (scale_factor)

        correlation = correlation.reshape(b, h, w, (2*self.r + 1)**2).permute(0, 3, 1, 2).contiguous()
        if final:
            v = x

            att = F.softmax(correlation, -1)  # [B, H*W, (2R+1)^2]

            v = (att.unsqueeze(-2)@v).squeeze(-2).reshape((b, h, w, 2)).permute(0, 3, 1, 2).contiguous() #B 2 H W
            return v, correlation

        return v, correlation

    def forward(self, f1, f2, c1, c0, final=False):
        f2 = self.warp(f2, c1)
        c1, correlation = self.corr_softmax(q=f1, k=f2, v=c1, c0=c0, final=final)

        return c1, correlation
