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
    torch.cuda.set_device(args.local_rank)
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    regularization = CosLoss(lambd=args.lambd).cuda(args.local_rank)
    criterions = {'criterion':criterion, 'regularization': regularization}
    model = models.__dict__[args.arch](num_classes=num_classes, rate=args.rate)
    if dist.get_rank==0:
        print(model)
        tb_logger = SummaryWriter(log_dir='runs/{}_{}'.format(args.arch, time.strftime('%Y-%m-%d %H-%M-%S'),
                                                              time.localtime()))
    else:
        tb_logger = None

    model.cuda(args.local_rank)
    optimizer, scheduler = prepare_other(model, args)
    interval = 0
    top1, _ = test(0, test_loader, model, criterions, tb_logger, args=args)
    print_rank0('----------------Model Acc {}%----------------'.format(top1))
    for epoch in range(0, args.epoch):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        begin = time.time()
        print_rank0('\n==>>[Epoch={:03d}/{:03d}] [learning_rate={:6.4f}] [Time={:.2f}s]'.format(epoch + 1, args.epochs,
                                                                                                     optimizer.state_dict()[
                                                                                                         'param_groups'][0][
                                                                                                         'lr'],
                                                                                                     interval) \
                         + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                            100 - recorder.max_accuracy(False)))
        train_acc1, train_los1 = train(epoch, train_loader, model, criterions, optimizer, tb_logger, args=args)
        # plot_similarity(model, epoch, args.arch)
        test_top1_2, test_los_2 = test(epoch, test_loader, model, criterions, tb_logger, args=args)
        merge(model, l, epoch, arr)

        is_best = recorder.update(epoch, train_los1, train_acc1, test_los_2, test_top1_2)
        interval = time.time() - begin

        if is_best:
            if dist.get_rank()==0:
                torch.save(model.cuda(0), 'best_{}_x3.pt'.format(args.arch))

    if dist.get_rank() == 0:
        plot_interval(arr, args.epochs)

if __name__ == '__main__':
    main()