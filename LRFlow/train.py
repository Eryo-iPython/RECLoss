import sys
sys.path.append('core')
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from core.utils.flow_viz import flow_to_image
from matplotlib import pyplot as plt
from core.lrflow import LRFlow
import argparse
from core.datasets import *
from torch.cuda.amp import GradScaler
from inference import InputPadder
import numpy as np
import time
import torch
from core.fundamentalmatrix import fun_loss, get_matrix
from core.utils.utils import coords_grid, random_sample
import datetime
from evalu import validate_chairs, validate_sintel, validate_kitti

MAX_FLOW = 400

def epe_loss(flow, target, valid, gamma=0.8):
    flowlen = len(flow)
    loss = 0.

    mag = (target ** 2).sum(dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < MAX_FLOW)

    for i in range(flowlen):
        i_weight = gamma ** (flowlen - i - 1)
        i_loss = torch.norm(flow[i]-target, p=2, dim=1)
        loss += i_weight * (valid * i_loss).mean()

    # epe = torch.sum((flow[-1] - target) ** 2, dim=1).sqrt()
    epe = i_loss.view(-1)[valid.view(-1)]

    return loss, epe.mean().item()

def seq_lossfc(args, flow, target, valid, gamma=0.8):
    flowlen = len(flow)
    loss = 0.

    mag = (target ** 2).sum(dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < MAX_FLOW)

    if args.funloss:
        if args.random:
            p1, l_target, l_flow = random_sample(args, target, flow)
        F, p1 = get_matrix(p1, l_target.to(p1.device))
        for i in range(flowlen):
            i_weight = gamma**(flowlen - i - 1)
            # if i == flowlen - 1:
            #     if args.random:
            #         f_loss = fun_loss(p1, l_target.to(p1.device), l_flow[i].to(p1.device))
            #     else:
            #         f_loss = fun_loss(p1, target[:, :, :10, :10].to(p1.device), flow[i][:, :, :10, :10].to(p1.device))
            # else:
            #     f_loss = torch.tensor(0.)

            if args.random:
                f_loss = fun_loss(p1, F, l_flow[i].to(p1.device))

            i_loss = (flow[i]-target).abs() + f_loss.abs()

            loss += i_weight * (valid[:, None] * i_loss).mean()
    else:
        for i in range(flowlen):
            i_weight = gamma ** (flowlen - i - 1)
            i_loss = (flow[i] - target).abs()

            loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.norm(flow[-1]-target, p=2, dim=1)
    epe = epe.view(-1)[valid.view(-1)]

    return loss, epe.mean().item()

def count_parameter(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


class Logger:
    def __init__(self, time=100):
        self.count = 0
        self.num = []
        self.time = time

    def get_epe(self, epe, speed):
        self.num.append(epe)
        self.count += 1
        if self.count % self.time == 0:
            print(f'--{self.count:06d}--  epe: {np.mean(self.num):.4f}  speed: {speed:.4f}s')
            self.num = []

def summery(file, data):
    with open(file, 'a+') as f:
        f.write(f'{data}')
        f.write('\n')
        f.close()

def main(args):
    torch.manual_seed(1234)
    np.random.seed(1234)

    if torch.cuda.is_available():
        model = nn.DataParallel(LRFlow(args=args), device_ids=args.device_ids)
        if args.ckpt is not None:
            print(r'Using pretrain model')
            model.load_state_dict(torch.load(args.ckpt), strict=False)
        model.cuda()
    else:
        model = LRFlow(args=args)
    model.train()
    if args.freeze_bn:
        model.module.freeze_bn()


    print('paramters:', count_parameter(model))
    print('add_noise:', args.add_noise)
    print('mixed_precision:', args.mixed_precision)
    # com_transform =flow_transforms.Compose([
    #     flow_transforms.CenterCrop(args.crop_size),
    # ])

    train_loader = fetch_dataloader(args)

    logger = Logger(time=100)

    optimizer = optim.AdamW(params=model.parameters(),
                            lr=args.lr,
                            weight_decay=args.wdecay,
                            eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.steps + 100,
                                             pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    print('Data ready')
    scaler = GradScaler(enabled=args.mixed_precision)

    print(f'cuda:{torch.cuda.is_available()}')
    print('Running..........')

    count = 0
    keep_train = True

    VAL_FREQ = 5000

    while keep_train:
        loss_avg = 0.
        epe_avg = 0.
        c = 0
        start = time.perf_counter()
        for i, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape)).clamp(0.0, 255.0)

            s = time.perf_counter()
            out = model(image1, image2, iters=args.iters)

            loss, epe = seq_lossfc(args, out, flow, valid, args.gamma)
            # loss, epe = epe_loss(out, flow, valid, args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            e = time.perf_counter()

            loss_avg += loss.item()
            epe_avg += epe
            c += 1

            logger.get_epe(epe, speed=(e-s)/args.batch_size)

            if count % VAL_FREQ == VAL_FREQ - 1:
                PATH = f'{count:06d}_{args.stage}.pth'
                torch.save(model.state_dict(), PATH)

                results = {'count': count}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(validate_chairs(model.module, iters=12))
                        summery('val_chairs.txt', results)
                        results = {}
                    elif val_dataset == 'sintel':
                        results.update(validate_sintel(model.module, iters=12))
                        summery('val_sintel.txt', results)
                        results = {}
                    elif val_dataset == 'kitti':
                        results.update(validate_kitti(model.module, iters=12))
                        summery('val_kitti.txt', results)
                        results = {}

                model.train()
                if args.freeze_bn:
                    model.module.freeze_bn()

            count += 1

            if count > args.steps:
                keep_train = False
                break

        end = time.perf_counter()
        print(f'time:---{datetime.datetime.now()}---')
        print(f'speed:{(end-start)/(c*args.batch_size)}')
        print([f'step:{count:06d}', f'loss:{loss_avg/c:.4f}', f'epe:{epe_avg/c:.4f}'])

        with open('log.txt', 'a+') as file:
            file.write(f'{count}: {loss_avg/c:.4f}; epe:{epe_avg/c:.4f}')
            file.write('\n')
            file.close()


    PATH = f'final_{args.stage}.pth'
    torch.save(model.state_dict(), PATH)


    return PATH

if __name__ == '__main__':
    r"""
    FlyingChairs_release\data
    MPI\training
    KITTI\training
    flything3d\
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--image_size', nargs='+', type=int, default=(288, 960))
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--stage', help='data:chairs,things,sintel,kitti', default='kitti')
    parser.add_argument('--ckpt', default='lrflow_sintel.pth')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--small', default=False)
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--convfeature', default=True)
    parser.add_argument('--funloss', default=True)
    parser.add_argument('--validation', type=str, nargs='+', default=['kitti'])
    parser.add_argument('--shift', default=False)
    parser.add_argument('--random', default=True)
    parser.add_argument('--point_nums', help='how many points to compute fmatrix: default**2' ,default=50)
    parser.add_argument('--freeze_bn', default=True)
    parser.add_argument('--encoder', help='0:conv, 1:mixed, 2:shift', default=0)
    parser.add_argument('--radius', default=1)

    args = parser.parse_args()

    main(args)
