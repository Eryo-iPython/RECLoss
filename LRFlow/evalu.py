import sys
sys.path.append('core')
import torch
import numpy as np
import os
from torch import nn
from core.lrflow import LRFlow
from core.utils.frame_utils import *
from core.utils.flow_viz import *
from torch.utils.data import DataLoader
from core.utils.utils import InputPadder
import glob
import datetime
import argparse
from core import datasets
from core.utils.utils import forward_interpolate

#root: MPI\test
@torch.no_grad()
def submi_sintel_vis(model, warm_start=False, iters=32, outpath=r"sintel_submission"):
    model.eval()

    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        flow_prev, sequence_prev = None, None
        print(len(test_dataset))
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(img1=image1, img2=image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            flow_img = flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(os.path.join('vis_sintel', dstype, sequence)):
                os.makedirs(os.path.join('vis_sintel', dstype, sequence))
            image.save(f'vis_sintel/{dstype}/{sequence}/{(frame+1):04d}.png')

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(outpath, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            writeFlow(output_file, flow)
            sequence_prev = sequence


#kitti\testing\image_2
@torch.no_grad()
def submi_kitti_vis(model, iters=24, outpath=r'kitti_submission'):
    model.eval()
    time = []

    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    for index in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[index]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, pre_flow = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(pre_flow[0]).permute(1, 2, 0).cpu().numpy()

        file_path = os.path.join(outpath, frame_id)

        writeFlowKITTI(file_path, flow)

        if not os.path.exists('vis_kitti/flow'):
            os.makedirs('vis_kitti/flow')

        flow_image = flow_to_image(flow)

        image = Image.fromarray(flow_image)

        image.save(f'vis_kitti/flow/{index:06d}_10.png')

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')
    out_list, epe_list = [], []
    print(len(val_dataset))
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision', default=False)
parser.add_argument('--convfeature', default=True)
parser.add_argument('--shift', default=False)
parser.add_argument('--encoder', default=0)
parser.add_argument('--radius', default=1)
args = parser.parse_args()

if __name__ == '__main__':

    model  = torch.nn.DataParallel(LRFlow(args=args), device_ids=[0, ])
    model.load_state_dict(torch.load('models/lrflow_kitti.pth'))
    model.cuda()
    model.eval()


    # submi_sintel_vis(model, warm_start=False)
    with torch.no_grad():
        # validate_sintel(model.module, iters=32)
        # submi_sintel_vis(model)
        validate_kitti(model.module, iters=12)

    # with torch.no_grad():
        # print(validate_kitti(model.module))
        # print(validate_chairs(model.module, iters=12))
        # print(validate_sintel(model.module))


