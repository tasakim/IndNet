import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
import math
from models.cifar.resnet import linear_conv3x3
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.cifar.custom_relu import C_ReLU
from models.cifar.resnet_v2 import ResNetBasicblock

class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def prepare_dataset(args):
    if args.dataset == 'cifar10':
        # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        # std = [x / 255 for x in [63.0, 62.1, 66.7]]
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = datasets.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = datasets.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_set = datasets.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        root = '/home/psdz/Downloads/imagenet'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'val')
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        train_set = datasets.ImageFolder(traindir, train_transform)
        test_set = datasets.ImageFolder(testdir, test_transform)
        num_classes = 1000

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    args.batch_size /= args.nprocs
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(args.batch_size), shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(args.batch_size), shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    return train_loader, test_loader, num_classes, train_sampler, test_sampler


def prepare_other(model, args):
    if 'cifar' in args.dataset:
        # param = [{'params': p for n, p in model.named_parameters() if 'shift' not in n}]
        # optimizer = torch.optim.SGD(param, args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1, last_epoch=-1)

    return optimizer, scheduler


def train(epoch, train_loader, model, criterions, optimizer, tb_logger, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model.train()
    num_iter = len(train_loader)
    criterion = criterions.get('criterion')
    regularization = criterions.get('regularization', None)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.cuda(args.local_rank)
        input = input.cuda(args.local_rank)

        adjust_learning_rate(optimizer, epoch, i, num_iter, args)
        output = model(input)
        loss = criterion(output, target)
        if regularization is not None:
            reg_loss = regularization(model)
            if dist.get_rank()==0:
                print(loss, reg_loss)
            loss += reg_loss

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_prec1 = reduce_mean(prec1, args.nprocs)
        reduced_prec5 = reduce_mean(prec5, args.nprocs)

        losses.update(reduced_loss.item(), input.size(0))
        top1.update(reduced_prec1.item(), input.size(0))
        top5.update(reduced_prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    if tb_logger is not None and dist.get_rank() == 0:
        tb_logger.add_scalar(tag='Loss/train', scalar_value=losses.avg, global_step=epoch)
        tb_logger.add_scalar(tag='Others/LR', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)
    return top1.avg, losses.avg


def test(epoch, test_loader, model, criterions, tb_logger, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    # model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model.eval()
    criterion = criterions.get('criterion')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.cuda(args.local_rank), target.cuda(args.local_rank)
            output = model(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_prec1 = reduce_mean(prec1, args.nprocs)
            reduced_prec5 = reduce_mean(prec5, args.nprocs)

            losses.update(reduced_loss.item(), input.size(0))
            top1.update(reduced_prec1.item(), input.size(0))
            top5.update(reduced_prec5.item(), input.size(0))

    print_rank0('**Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                 error1=100 - top1.avg))
    if tb_logger is not None and dist.get_rank() == 0:
        tb_logger.add_scalar(tag='Loss/test', scalar_value=losses.avg, global_step=epoch)

    return top1.avg, losses.avg


def setup_seed(seed=None):
    if seed is not None:
        manualSeed = seed
    else:
        manualSeed = random.randint(1, 100)

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


class RecorderMeter():
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        assert total_epoch > 0
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_losses = self.epoch_losses - 1

        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = self.epoch_accuracy

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        #    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
        self.epoch_losses[idx, 0] = train_loss
        self.epoch_losses[idx, 1] = val_loss
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = round(val_acc, 2)
        self.current_epoch = idx + 1
        # x = float(int(self.max_accuracy(False) * 1000) / 1000)
        x = self.max_accuracy(False)
        y = val_acc
        # y = float(val_acc * 1000) / 1000)
        return abs(y - x) * 100 <= 1

    def max_accuracy(self, istrain):
        if self.current_epoch <= 0: return 0
        if istrain:
            return self.epoch_accuracy[:self.current_epoch, 0].max()
        else:
            return self.epoch_accuracy[:self.current_epoch, 1].max()

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis * 50, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis * 50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('---- save figure {} into {}'.format(title, save_path))
        plt.close(fig)


def print_rank0(info):
    if dist.get_rank() == 0:
        print(info)

def adjust_learning_rate(optimizer, epoch, step, len_iter, args):

    if args.lr_type == 'step':
        factor = epoch // 125
        lr = args.lr * (0.1 ** factor)

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.5 ** factor)

    elif args.lr_type == 'cos':
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.lr

    else:
        raise NotImplementedError

    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def set_module_v2(model, submodule_key, key2, module):
    tokens = submodule_key.split('.')
    tokens.append(key2)
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def fuse_model(model, l):
    layer_score_id = 0
    conv_count = 1
    l1 = l['l1']
    l2 = l['l2']
    l3 = l['l3']
    skip = l['skip']
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, (linear_conv3x3, nn.Conv2d)):
            if conv_count in l1:
                coeffi = module.coeffi_matrix
                weight = module.weight_matrix
                num = weight.shape[0]
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=module.in_planes, out_channels=weight.shape[0],
                                      kernel_size=3, stride=module.stride,
                                      padding=module.padding, bias=module.bias is not None)
                new_layer.weight = nn.Parameter(weight.reshape(-1, module.in_planes, 3, 3))
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(model, name, new_layer)
                layer_score_id += 1
                conv_count += 1

            elif conv_count in l2:
                pass

            elif conv_count in l3:
                weight=  module.weight.data
                base_weight = weight[:, :num, :, :]
                extra_weight = weight[:, num:, :, :]
                shape = extra_weight.shape
                new_weight = torch.mm(coeffi.transpose(1, 0), extra_weight.permute(1, 0, 2, 3).flatten(1).reshape([-1, shape[0], shape[2], shape[3]]))
                weight = new_weight.permute(1, 0, 2, 3)
                weight += base_weight
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2], stride=module.stride,
                                      padding=module.padding, groups=module.groups, dilation=module.dilation,
                                      bias=module.bias is not None)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(model, name, new_layer)
                layer_score_id += 1
                conv_count += 1
            elif conv_count in skip:
                layer_score_id += 1
                conv_count += 1

            else:
                conv_count += 1

        elif isinstance(module, nn.BatchNorm2d):
            if conv_count -1 in l1+l2:
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                running_mean = torch.index_select(module.running_mean.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                running_var = torch.index_select(module.running_var.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                new_layer = nn.BatchNorm2d(num_features=num)
                new_layer.weight.data = nn.Parameter(weight)
                new_layer.bias.data = nn.Parameter(bias) if module.bias is not None else None
                new_layer.running_mean = running_mean
                new_layer.running_var = running_var
                set_module(model, name, new_layer)

        elif isinstance(module, C_ReLU):
            new_layer = nn.ReLU(inplace=True)
            set_module(model, name, new_layer)

    return model

def mask_model(model, l):
    layer_score_id = 0
    conv_count = 1
    l1 = l['l1']
    l2 = l['l2']
    l3 = l['l3']
    skip = l['skip']
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, (linear_conv3x3, nn.Conv2d)):
            if conv_count in l1:
                coeffi = module.coeffi_matrix
                weight = module.weight_matrix
                num = weight.shape[0]
                weight_zero = torch.zeros([coeffi.shape[0], weight.shape[1]]).to(device)
                weight = torch.cat([weight, weight.zero], dim=0)
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=module.in_planes, out_channels=module.out_planes,
                                      kernel_size=3, stride=module.stride,
                                      padding=module.padding, bias=module.bias is not None)
                new_layer.weight = nn.Parameter(weight.reshape(-1, module.in_planes, 3, 3))
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(model, name, new_layer)
                layer_score_id += 1
                conv_count += 1

            elif conv_count in l2:
                pass

            elif conv_count in l3:
                weight = module.weight.data
                base_weight = weight[:, :num, :, :]
                extra_weight = weight[:, num:, :, :]
                shape = extra_weight.shape
                new_weight = torch.mm(coeffi.transpose(1, 0), extra_weight.permute(1, 0, 2, 3).flatten(1).reshape(
                    [-1, shape[0], shape[2], shape[3]]))
                weight = new_weight.permute(1, 0, 2, 3)
                weight += base_weight
                weight_zero = torch.zeros_like(extra_weight).to(device)
                weight = torch.cat([weight, weight_zero], dim=1)
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2], stride=module.stride,
                                      padding=module.padding, groups=module.groups, dilation=module.dilation,
                                      bias=module.bias is not None)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(model, name, new_layer)
                layer_score_id += 1
                conv_count += 1
            elif conv_count in skip:
                layer_score_id += 1
                conv_count += 1

            else:
                conv_count += 1

        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                pass
                # weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                # bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                # running_mean = torch.index_select(module.running_mean.data, dim=0,
                #                                   index=torch.LongTensor(range(num)).to(device))
                # running_var = torch.index_select(module.running_var.data, dim=0,
                #                                  index=torch.LongTensor(range(num)).to(device))
                # new_layer = nn.BatchNorm2d(num_features=num)
                # new_layer.weight.data = nn.Parameter(weight)
                # new_layer.bias.data = nn.Parameter(bias) if module.bias is not None else None
                # new_layer.running_mean = running_mean
                # new_layer.running_var = running_var
                # set_module(model, name, new_layer)

        elif isinstance(module, C_ReLU):
            new_layer = nn.ReLU(inplace=True)
            set_module(model, name, new_layer)

    return model


def plot_similarity(model, epoch, arch):

    for name, param in model.named_parameters():
        if 'weight_matrix' in name:
            if not os.path.exists('./sim/{}/{}'.format(arch, name)):
                os.mkdir('./sim/{}/{}'.format(arch, name))
            sim = F.cosine_similarity(param.flatten(1).cpu().detach().unsqueeze(1), param.flatten(1).cpu().detach().unsqueeze(0), dim=2)
            plt.imshow(sim.numpy(), cmap='hot', interpolation='nearest', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(name + '/' + str(epoch))
            plt.savefig('./sim/{}/{}/{}_sim.png'.format(arch, name, epoch), dpi=300)
            plt.clf()



def fuse_model_v2(model, l=None):
    layer_score_id = 0
    conv_count = 1
    l1 = l['l1']
    l2 = l['l2']
    l3 = l['l3']
    skip = l['skip']
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, ResNetBasicblock):
            layer = module.conv2
            co_weight = module.weight.data
            num_base, num_extra = co_weight.shape
            conv2_weight = layer.weight.data

            extra_weight = torch.matmul(conv2_weight.flatten(2)[:, num_base:, :].permute(0, 2, 1),
                                        co_weight.permute(1, 0)).permute(0, 2, 1).reshape(conv2_weight.shape[0], -1,
                                                                                          conv2_weight.shape[2],
                                                                                          conv2_weight.shape[3])
            weight = conv2_weight[:, :num_base, ...] + extra_weight
            bias = layer.bias.data if layer.bias is not None else None
            new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                  kernel_size=weight.shape[2], stride=layer.stride,
                                  padding=layer.padding, groups=layer.groups, dilation=layer.dilation,
                                  bias=layer.bias)
            new_layer.weight = nn.Parameter(weight)
            new_layer.bias = nn.Parameter(bias) if layer.bias is not None else None
            set_module_v2(model, name, 'conv2', new_layer)
            set_module_v2(model, name, 'weight', None)

    return model

def merge(model, l, epoch, arr):
    layer_score_id = 0
    conv_count = 1
    l1 = l['l1']
    l2 = l['l2']
    l3 = l['l3']
    skip = l['skip']
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if conv_count in l1:
                param = module.weight
                sim = F.cosine_similarity(param.flatten(1).cpu().detach().unsqueeze(1),
                                          param.flatten(1).cpu().detach().unsqueeze(0), dim=2).abs().sum(dim=1)
                if epoch==0:
                    arr[str(conv_count)] = sim.sort().values().numpy()
                else:
                    arr[str(conv_count)] = np.append(arr[str(conv_count)], sim.sort().values.numpy())
                layer_score_id += 1
                conv_count += 1

            elif conv_count in l2:
                pass

            elif conv_count in l3:
                layer_score_id += 1
                conv_count += 1
            elif conv_count in skip:
                layer_score_id += 1
                conv_count += 1

            else:
                conv_count += 1

        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                pass

    return model



class CosLoss(nn.Module):

    def __init__(self, lambd):
        self.lambd = lambd


    def forward(self, model):
        loss = 0.
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, module in model.named_modules():
            if isinstance(module, ResNetBasicblock):
                w = module.conv1.weight.data.flatten(1)
                device = w.device
                tmp = torch.mm(w, w.t()).exp()
                tmp_sum = tmp.sum(dim=-1)
                l = (tmp * torch.eye(w.shape[0], device=device)).sum(dim=-1)
                loss += (l/tmp_sum).sum()
            else:
                pass
        return loss * self.lambd
