import sys
sys.path.append('core')
import torch
from torch.nn import functional as F
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from core.utils.flow_viz import flow_to_image
from core.raft import RAFT
import glob
import os

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def get_img(file_name, device):
    img = np.array(Image.open(file_name)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def draw(img, flow, groud_truth):

    img = img.permute(0, 2, 3, 1).contiguous()
    groud_truth = groud_truth.permute(0, 2, 3, 1).contiguous()

    img = img[0].cpu().detach().numpy()
    groud_truth = groud_truth[0].cpu().detach().numpy()

    flow = flow_to_image(flow)
    groud_truth = flow_to_image(groud_truth)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(flow)
    plt.subplot(223)
    plt.imshow(groud_truth)
    plt.show()


@torch.no_grad()
def save_flow_2_image(image_dir, model, device,iters=20):
    images = glob.glob(os.path.join(image_dir, '*.png')) + \
             glob.glob(os.path.join(image_dir, '*.jpg'))

    images = sorted(images)
    # model = torch.nn.DataParallel(model)
    # model = model.module
    model.to(device)
    model.eval()

    if not os.path.exists(os.path.join(image_dir, 'flow')):
        os.makedirs(os.path.join(image_dir, 'flow'))

    with torch.no_grad():
        for index, (frame1, frame2) in enumerate(zip(images[:-1], images[1:])):

            frame1 = get_img(frame1, device)
            frame2 = get_img(frame2, device)

            padder = InputPadder(frame1.shape)
            frame1, frame2 = padder.pad(frame1, frame2)
            print(frame1.shape)
            flow_pre = model(frame1, frame2, iters=iters)
            flow = flow_pre[-1]

            flow_up = padder.unpad(flow[0]).permute(1, 2, 0).cpu().numpy()

            flow_image = flow_to_image(flow_up)

            image = Image.fromarray(flow_image)
            image.save(f'{image_dir}/flow/{index + 1:04d}.png')


if __name__ == '__main__':
    model = Rot()
    save_flow_2_image(r'us', model, device='cpu', iters=20)
