import random
import numpy
import torch
import os
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
import numpy as np
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
        root = args.data_path
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
        # optimizer = torch.optim.SGD(param, args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
                                                         gamma=0.1, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.99, weight_decay=1e-4, nesterov=True)
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
    regularization2 = criterions.get('regularization2', None)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.cuda(args.local_rank)
        input = input.cuda(args.local_rank)

        adjust_learning_rate(optimizer, epoch, i, num_iter, args)
        output = model(input)
        loss = criterion(output, target)
        if regularization is not None:
            reg_loss = regularization(model)
            if dist.get_rank() == 0 and epoch == 0:
                print(loss, reg_loss)
            loss += reg_loss

        if regularization2 is not None:
            reg_loss2 = regularization2(model)
            if dist.get_rank() == 0 and epoch == 0:
                print(reg_loss2)
            loss += reg_loss2

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
        # print('writing tensorboard...')
    return top1.avg, top5.avg


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
    torch.manual_seed(manualSeed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(manualSeed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(manualSeed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(manualSeed)  # Numpy module.
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


def print_rank0(info):
    if dist.get_rank() == 0:
        print(info)


def adjust_learning_rate(optimizer, epoch, step, len_iter, args):
    if args.lr_type == 'step':
        factor = epoch // 125
        # if epoch >= 80:
        #     factor = factor + 1
        lr = args.lr * (0.1 ** factor)

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError

    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # if step == 0:
    #     print_rank0('learning_rate: ' + str(lr))


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
                                      kernel_size=3,
                                      stride=module.stride, padding=module.padding, bias=module.bias is not None)
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
                new_weight = torch.mm(coeffi.transpose(1, 0), extra_weight.permute(1, 0, 2, 3).flatten(1)).reshape(
                    [-1, shape[0], shape[2], shape[3]])
                weight = new_weight.permute(1, 0, 2, 3)
                weight += base_weight
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(model, name, new_layer)
                conv_count += 1
                layer_score_id += 1

            elif conv_count in skip:
                conv_count += 1
                layer_score_id += 1

            else:
                conv_count += 1

        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                running_mean = torch.index_select(module.running_mean.data, dim=0,
                                                  index=torch.LongTensor(range(num)).to(device))
                running_var = torch.index_select(module.running_var.data, dim=0,
                                                 index=torch.LongTensor(range(num)).to(device))
                new_layer = nn.BatchNorm2d(num_features=num)
                new_layer.weight.data = nn.Parameter(weight)
                new_layer.bias.data = nn.Parameter(bias) if module.bias is not None else None
                new_layer.running_mean = running_mean
                new_layer.running_var = running_var
                set_module(model, name, new_layer)

        elif isinstance(module, C_ReLU):
            new_layer = nn.ReLU()
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
                weight = torch.cat([weight, weight_zero], dim=0)
                # mask = torch.zeros([coeffi.shape[0] + coeffi.shape[1], 1, 1, 1])
                # mask = mask.index_fill_(dim=0, index=torch.arange(coeffi.shape[1]), value=1)

                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=module.in_planes, out_channels=module.out_planes,
                                      kernel_size=3,
                                      stride=module.stride, padding=module.padding, bias=module.bias is not None)
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
                new_weight = torch.mm(coeffi.transpose(1, 0), extra_weight.permute(1, 0, 2, 3).flatten(1)).reshape(
                    [-1, shape[0], shape[2], shape[3]])
                weight = new_weight.permute(1, 0, 2, 3)
                weight += base_weight
                weight_zero = torch.zeros_like(extra_weight).to(device)
                weight = torch.cat([weight, weight_zero], dim=1)
                bias = module.bias.data if module.bias is not None else None
                new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                                      kernel_size=weight.shape[2],
                                      stride=module.stride, padding=module.padding, groups=module.groups,
                                      dilation=module.dilation, bias=module.bias)
                new_layer.weight = nn.Parameter(weight)
                new_layer.bias = nn.Parameter(bias) if module.bias is not None else None
                set_module(model, name, new_layer)
                conv_count += 1
                layer_score_id += 1

            elif conv_count in skip:
                conv_count += 1
                layer_score_id += 1

            else:
                conv_count += 1

        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                pass
                # weight = torch.index_select(module.weight.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                # bias = torch.index_select(module.bias.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                # running_mean = torch.index_select(module.running_mean.data, dim=0,
                #                                   index=torch.LongTensor(range(num)).to(device))
                # running_var = torch.index_select(module.running_var.data, dim=0, index=torch.LongTensor(range(num)).to(device))
                # new_layer = nn.BatchNorm2d(num_features=num)
                # new_layer.weight.data = nn.Parameter(weight)
                # new_layer.bias.data = nn.Parameter(bias) if module.bias is not None else None
                # new_layer.running_mean = running_mean
                # new_layer.running_var = running_var
                # set_module(model, name, new_layer)
    return model


def plot_similarity(model, epoch, arch):
    for name, param in model.named_parameters():
        if 'weight_matrix' in name:
            if not os.path.exists('./sim/{}/{}'.format(arch, name)):
                os.mkdir('./sim/{}/{}'.format(arch, name))
            sim = F.cosine_similarity(param.flatten(1).cpu().detach().unsqueeze(1),
                                      param.flatten(1).cpu().detach().unsqueeze(0), dim=2)
            plt.imshow(sim.numpy(), cmap='hot', interpolation='nearest', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(name + '/' + str(epoch))
            plt.savefig('./sim/{}/{}/{}_sim.png'.format(arch, name, epoch), dpi=300)
            plt.clf()


def fuse_model_v2(model, l=None):
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
                                  kernel_size=weight.shape[2],
                                  stride=layer.stride, padding=layer.padding, groups=layer.groups,
                                  dilation=layer.dilation, bias=layer.bias)
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
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if conv_count in l1:
                param = module.weight
                sim = F.cosine_similarity(param.flatten(1).cpu().detach().unsqueeze(1),
                                          param.flatten(1).cpu().detach().unsqueeze(0), dim=2).abs().sum(dim=1)
                # if dist.get_rank() == 0 and conv_count == l1[0]:
                #     print(sim.sort().values)
                if epoch == 0:
                    arr[str(conv_count)] = sim.sort().values.numpy()
                else:
                    arr[str(conv_count)] = numpy.append(arr[str(conv_count)], sim.sort().values.numpy())
                layer_score_id += 1
                conv_count += 1

            elif conv_count in l2:
                pass

            elif conv_count in l3:
                conv_count += 1
                layer_score_id += 1

            elif conv_count in skip:
                conv_count += 1
                layer_score_id += 1

            else:
                conv_count += 1

        elif isinstance(module, nn.BatchNorm2d):
            if conv_count - 1 in l1 + l2:
                pass

    return model


# class CosLoss(nn.Module):
#
#     def __init__(self, lambd):
#         super(CosLoss, self).__init__()
#         self.lambd = lambd
#
#     def forward(self, model):
#         loss = 0.
#         model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
#         for name, module in model.named_modules():
#             if isinstance(module, ResNetBasicblock):
#                 w = module.conv1.weight.data.flatten(1)
#                 device = w.device
#                 tmp = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=-1) - torch.eye(w.shape[0], device=device)
#                 loss += tmp.abs().sum()
#             else:
#                 pass
#         return loss * self.lambd


# class CosLoss(nn.Module):
#
#     def __init__(self, lambd):
#         super(CosLoss, self).__init__()
#         self.lambd = lambd
#
#     def forward(self, model):
#         loss = 0.
#         model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
#         for name, module in model.named_modules():
#             if isinstance(module, ResNetBasicblock):
#                 w = module.conv1.weight.data.flatten(1)
#                 device = w.device
#                 tmp = torch.mm(w, w.t()).exp()
#                 tmp_sum = tmp.sum(dim=-1)
#                 l = (tmp * (1 - torch.eye(w.shape[0], device=device))).sum(dim=-1)
#                 loss += (l/tmp_sum).sum()
#             else:
#                 pass
#         return loss * self.lambd

class BaseLoss(nn.Module):

    def __init__(self, lambd):
        super(BaseLoss, self).__init__()
        self.lambd = lambd

    def forward(self, model):
        loss = 0.
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, module in model.named_modules():
            if isinstance(module, ResNetBasicblock):
                w = module.conv1.weight.flatten(1)
                device = w.device
                logits = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=-1)
                labels = torch.arange(w.shape[0], device=device)
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.t(), labels)
                loss += (loss_t + loss_i) / 2
            else:
                pass
        return loss * self.lambd

class ExtraLoss(nn.Module):

    def __init__(self, lambd):
        super(ExtraLoss, self).__init__()
        self.lambd = lambd

    def forward(self, model):
        loss = 0.
        model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        for name, module in model.named_modules():
            if isinstance(module, ResNetBasicblock):
                w = module.weight
                device = w.device
                logits = F.cosine_similarity(w.unsqueeze(1), w.unsqueeze(0), dim=-1)
                labels = torch.arange(w.shape[0], device=device)
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.t(), labels)
                loss += (loss_t + loss_i) / 2
            else:
                pass
        return loss * self.lambd


def transform(epoch, model, args):
    # cur_rate = args.rate - (args.rate - args.compress_rate) * epoch / args.epochs
    cur_rate = 0.9 - (0.9 - 0.6) * epoch / 400
    for name, module in model.named_modules():
        if isinstance(module, ResNetBasicblock):
            layer1 = module.conv1
            coeffi = module.weight
            conv_weight = layer1.weight.data
            bias = layer1.bias.data if layer1.bias is not None else None
            keep_num = int(cur_rate*module.planes)
            device = conv_weight.device
            if conv_weight.shape[0] - keep_num >=1:
                index = F.cosine_similarity(conv_weight.flatten(1).unsqueeze(1), conv_weight.flatten(1).unsqueeze(0), dim=-1).sum(dim=1).topk(conv_weight.shape[0] - keep_num).indices
                index = index.sort().values
                target = conv_weight[index, :]
                mask = torch.ones(conv_weight.size(0), dtype=torch.bool, device=device)
                mask[index] = False
                new_conv_weight = conv_weight[mask, ...]
                coefficients = []
                for i, row in enumerate(target):
                    row = row.flatten()
                    matrix = new_conv_weight.flatten(1).t()
                    coefficient = torch.linalg.lstsq(matrix, row).solution.unsqueeze(-1)
                    coefficients.append(coefficient)
                    coeffi[mask, :].data += coeffi[index[i], :].data * coefficient
                coefficients = torch.cat(coefficients, dim=-1)

                new_layer = nn.Conv2d(in_channels=new_conv_weight.shape[1], out_channels=new_conv_weight.shape[0],
                                      kernel_size=new_conv_weight.shape[2],
                                      stride=layer1.stride, padding=layer1.padding, groups=layer1.groups,
                                      dilation=layer1.dilation, bias=True if layer1.bias is not None else False)
                new_layer.weight = nn.Parameter(new_conv_weight)

                new_layer.bias = nn.Parameter(bias) if layer1.bias is not None else None

                set_module_v2(model, name, 'conv1', new_layer)

                new_weight = nn.Parameter(torch.cat([coeffi[mask, :], coefficients], dim=-1))

                set_module_v2(model, name, 'weight', new_weight)

                layer_bn = module.bn1
                weight = layer_bn.weight.data[mask]
                bias = layer_bn.bias.data[mask]
                running_mean = layer_bn.running_mean.data[mask]
                running_var = layer_bn.running_var.data[mask]
                new_layer = nn.BatchNorm2d(num_features=new_conv_weight.shape[0])
                new_layer.weight.data = nn.Parameter(weight)
                new_layer.bias.data = nn.Parameter(bias) if layer_bn.bias is not None else None
                new_layer.running_mean = running_mean
                new_layer.running_var = running_var
                set_module_v2(model, name, 'bn1', new_layer)

                layer2 = module.conv2
                mask2 = torch.cat([mask, torch.ones([module.planes - len(mask)], dtype=torch.bool, device=device)], dim=0)
                new_conv_weight = torch.concat([layer2.weight.data[:, mask2, ...], layer2.weight.data[:, ~mask2, ...]], dim=1)
                bias = layer2.bias.data if layer2.bias is not None else None
                new_layer = nn.Conv2d(in_channels=new_conv_weight.shape[1], out_channels=new_conv_weight.shape[0],
                                     kernel_size=new_conv_weight.shape[2],
                                     stride=layer2.stride, padding=layer2.padding, groups=layer2.groups,
                                     dilation=layer2.dilation, bias=True if layer2.bias is not None else False)
                new_layer.weight = nn.Parameter(new_conv_weight)

                new_layer.bias = nn.Parameter(bias) if layer2.bias is not None else None

                set_module_v2(model, name, 'conv2', new_layer)
            else:
                pass

# if __name__ == '__main__':
#     from models.cifar.resnet_v2 import resnet56_v2, ResNetBasicblock
#
#     model = resnet56_v2(10, 0.25)
#     l1 = [2, 4, 6, 8, 11, 13, 15, 18, 20]  # r20
#     l2 = []
#     l3 = [3, 5, 7, 9, 12, 14, 16, 19, 21]
#     skip = [10, 17]
#     layer_score_id = 0
#     conv_count = 1
#     for name, module in model.named_modules():
#         if isinstance(module, ResNetBasicblock):
#             layer = module.conv2
#             co_weight = module.weight.data
#             num_base, num_extra = co_weight.shape
#             conv2_weight = layer.weight.data
#
#             extra_weight = torch.matmul(conv2_weight.flatten(2)[:, num_base:, :].permute(0, 2, 1),
#                                         co_weight.permute(1, 0)).permute(0, 2, 1).reshape(conv2_weight.shape[0], -1,
#                                                                                           conv2_weight.shape[2],
#                                                                                           conv2_weight.shape[3])
#             weight = conv2_weight[:, :num_base, ...] + extra_weight
#             bias = layer.bias.data if layer.bias is not None else None
#             new_layer = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
#                                   kernel_size=weight.shape[2],
#                                   stride=layer.stride, padding=layer.padding, groups=layer.groups,
#                                   dilation=layer.dilation, bias=layer.bias)
#             new_layer.weight = nn.Parameter(weight)
#             new_layer.bias = nn.Parameter(bias) if layer.bias is not None else None
#
#             set_module_v2(model, name, 'conv2', new_layer)
#             set_module_v2(model, name, 'weight', None)
#
#     print(model)
#     model = model.eval()
#     x = torch.randn([1, 3, 32, 32])
#     y = model(x)


if __name__ == '__main__':
    from models.cifar.resnet_v2 import resnet56_v2, ResNetBasicblock
    model = resnet56_v2(10, 0.9)
    transform(300, model, None)
    print(model)

