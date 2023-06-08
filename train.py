import argparse

import torch.cuda
from utils import *
import time
import models
import sys
import os
from visual import plot_interval
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append((current_dir))

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--dataset', type=str, default='cifar10',)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_type', type=str, default='cos')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--arch', type=str, default='resnet56_v2')
parser.add_argument('--rate', type=float, default=0.5)
parser.add_argument('--lambd', type=float, default=1e-5)

parser.add_argument('--checkpoint', type=str)
parser.add_argument('--save_path', type=str, default='/')
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()

print(args)

l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
l2 = []
l3 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57]
skip = [22, 41]
l = {
    'l1': l1,
    'l2': l2,
    'l3': l3,
    'skip': skip}

arr = {}

def main():
    setup_seed(42)
    recorder = RecorderMeter(args.epochs)
    dist.init_process_group(backend='nccl')




if __name__ == '__main__':
    main()