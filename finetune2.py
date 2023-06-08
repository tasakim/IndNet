import argparse

import torch

from utils import *
import time
from models import *
from torch.cuda.amp import GradScaler
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--dataset', type=str, default='cifar10', )
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_type', type=str, default='cos')
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--checkpoint', type=str)
parser.add_argument('--save_path', type=str, default='/')
parser.add_argument('--local_rank', type=int, default=-1)
args = parser.parse_args()
args.nprocs = torch.cuda.device_count()

print(args)

if args.dataset == 'cifar10':
    args.epochs = 400
else:
    args.epochs = 180

# l1 = [2, 4, 6, 8, 11, 13, 15, 18, 20] #r20
# l2 = []
# l3 = [3, 5, 7, 9, 12, 14, 16, 19]
# skip = [10, 17]

l1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]  # r56
l2 = []
l3 = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57]
skip = [22, 41]
l = {
    'l1': l1,
    'l2': l2,
    'l3': l3,
    'skip': skip}


def main():
    setup_seed(42)
    recorder = RecorderMeter(args.epochs)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    train_loader, test_loader, num_classes, train_sampler, test_sampler = prepare_dataset(args)
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    regularization = None
    criterions = {'criterion': criterion, 'regularization': regularization}
    ori_model = torch.load(args.checkpoint)
    pruned_model = deepcopy(ori_model)

    ori_model.cuda(args.local_rank)
    print_rank0('---------------Test Original Model---------------')
    top1, top5 = test(0, test_loader, ori_model, criterions, tb_logger=None, args=args)
    print_rank0('---------------Original Model Acc {}|{}%---------------'.format(top1, top5))
    del ori_model

    pruned_model.cuda(args.local_rank)
    pruned_model = fuse_model(pruned_model, l)
    print_rank0('---------------Test Pruned Model---------------')
    top1, top5 = test(0, test_loader, pruned_model, criterions, tb_logger=None, args=args)
    print_rank0('---------------Pruned Model Acc {}|{}%---------------'.format(top1, top5))


    if dist.get_rank == 0:
        print(pruned_model)
    pruned_model.cuda(args.local_rank)
    optimizer, scheduler = prepare_other(pruned_model, args)
    interval = 0
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
        train_acc1, train_los1 = train(epoch, train_loader, pruned_model, criterions, optimizer, tb_logger=None,
                                       args=args)

        test_top1_2, test_los_2 = test(epoch, test_loader, pruned_model, criterions, tb_logger=None, args=args)

        # scheduler.step()

        is_best = recorder.update(epoch, train_los1, train_acc1, test_los_2, test_top1_2)
        interval = time.time() - begin


if __name__ == '__main__':
    main()