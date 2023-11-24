from extractor import BlockEncoder
from update import UpdateBlock
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from cmm import CorrelationMatchingModule

autocast = torch.cuda.amp.autocast


def warp(img, vgrid):
#
    H, W = img.shape[-2:]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
#
    vgrid = vgrid.permute(0, 2, 3, 1).contiguous()
#
    img = F.grid_sample(img, vgrid, align_corners=True)
#
    return img

def init_grid(img):
    B, _, H, W = img.shape
    xx = torch.arange(W, device=img.device)
    yy = torch.arange(H, device=img.device)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    grid = coords[None].repeat(B, 1, 1, 1)

    return grid

class LRFlow(nn.Module):
    def __init__(self, outdim=128, h_dim=128, dropout=0., args=None):
        super(LRFlow, self).__init__()
        if args.encoder == 0:
            self.encoder = BlockEncoder(outc=outdim, dropout=dropout)

        self.update = UpdateBlock(h_dim, args=args)

        self.cmm = CorrelationMatchingModule(args.radius)
        self.args = args


    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def forward(self, img1, img2, test_mode=False, iters=12, flow_init=None):

        img1 = 2 * (img1 / 255.0) - 1.0
        img2 = 2 * (img2 / 255.0) - 1.0

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        with autocast(self.args.mixed_precision):
            f1, f2 = self.encoder([img1, img2])

        f1 = f1.float()
        f2 = f2.float()

        c0 = init_grid(f1)
        c1 = init_grid(f1)

        flow_list = []
        net = f1

        if flow_init is not None:
            c1 = c1 + flow_init

        for i in range(iters):
            c1 = c1.detach()

            c1, correlation = self.cmm(f1, f2, c1, c0, final=False)
            flow = c1 - c0

            with autocast(self.args.mixed_precision):
                net, mask, delta_flow = self.update(net, correlation, flow)

            c1 = c1 + delta_flow

            flow_up = self.upsample_flow(c1 - c0, mask)

            flow_list.append(flow_up)

        if test_mode:
            return c1 - c0, flow_up

        return flow_list
