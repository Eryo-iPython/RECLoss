import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid
import numpy as np

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


def window_partition_(tensor, window_size=(4, 4)):
    b, c, ph, pw = tensor.shape
    wh, ww = window_size

    window = tensor.reshape((b, c, ph//wh, wh, pw//ww, ww))
    window = window.permute(0, 2, 4, 1, 3, 5).contiguous()
    window = window.reshape((-1, c, wh, ww))

    return window

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)

        self.corr_pyramid.append(corr) #金字塔
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i

            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())
    #
    # @staticmethod
    # def corr(fmap1, fmap2, window_size=(4, 4)):
    #     B, C, H, W = fmap1.shape
    #     wh, ww = window_size
    #     ph, pw = H, W
    #
    #     if H % wh != 0 or W % ww != 0:
    #         b = (wh - H % wh) % wh
    #         r = (ww - W % ww) % ww
    #
    #         fmap1 = F.pad(fmap1, [0, r, 0, b, 0, 0])
    #         fmap2 = F.pad(fmap2, [0, r, 0, b, 0, 0])
    #
    #         _, _, ph, pw = fmap1.shape
    #
    #     matrix = []
    #     window_nums = ph//wh * pw//ww
    #
    #     fmap1 = window_partition_(fmap1, window_size)
    #     fmap1 = fmap1.reshape((B, window_nums, C, wh, ww))
    #
    #     for n in range(window_nums):
    #         for i in range(0, ph, wh):
    #             for j in range(0, pw, ww):
    #                 x = fmap1[:, n]
    #                 y = fmap2[:, :, i:i+wh, j:j+ww]
    #                 x = x.reshape((B, C, -1))
    #                 y = y.reshape((B, C, -1))
    #                 corr = torch.matmul(x.transpose(-1, -2), y)
    #                 corr = corr.cpu().detach().numpy()
    #                 matrix.append(corr)
    #
    #     matrix = np.array(matrix)
    #     matrix = torch.from_numpy(matrix)
    #     matrix = matrix.permute(1, 0, 2, 3).contiguous()
    #
    #     matrix = matrix.reshape((B, ph//wh, pw//ww, ph//wh, pw//ww, wh, ww, wh, ww))
    #     matrix = matrix.permute(0, 1, 2, 5, 6, 3, 4, 7, 8).contiguous()
    #     matrix = matrix.permute(0, 1, 3, 2, 4, 5, 7, 6, 8).contiguous()
    #     matrix = matrix.reshape((B, ph, pw, ph, pw))
    #     matrix = matrix[:, :H, :W, :H, :W]
    #     matrix = matrix.reshape((B, H, W, 1, H, W))
    #     matrix = matrix.to(fmap1.device)
    #
    #     return matrix / torch.sqrt(torch.tensor(C).float())




