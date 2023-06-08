import torch.nn as nn
import torch.nn.functional as F
import torch

expand_ratio = 3

class linear_conv3x3(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, rate=1.0):
        super(linear_conv3x3, self).__init__()
        num = int(out_planes * rate)

        self.coeffi_matrix = nn.Parameter(torch.randn([expand_ratio * out_planes - num, num]), requires_grad=True)
        self.weight_matrix = nn.Parameter(torch.randn([num, in_planes * 3 * 3]), requires_grad=True)
        self.bias = None
        self.out_planes = out_planes
        self.in_planes = in_planes
        self.stride = stride
        self.padding = 1

    def forward(self, x):
        self.weight = torch.mm(self.coeffi_matrix, self.weight_matrix)
        self.weight = torch.cat([self.weight_matrix, self.weight], dim=0)
        self.weight = self.weight.reshape([expand_ratio * self.out_planes, self.in_planes, 3, 3])
        out = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1.0):
        super(ResNetBasicblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, int(rate * planes), kernel_size=3, stride=stride, padding=1)
        self.weight = nn.Parameter(torch.randn([int(rate * planes), int((expand_ratio-rate)*planes)]), requires_grad=True)
        self.bn1 = nn.BatchNorm2d(int(rate * planes))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(expand_ratio * planes), planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        shape = out.shape
        tmp = torch.matmul(out.flatten(2).permute(0, 2, 1),self.weight).permute(0, 2, 1).reshape(shape[0], -1, shape[2], shape[3])
        out = torch.cat([out, tmp], dim=1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, rate=1.0):
        self.inplanes = 16
        super(CifarResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], rate=rate)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, rate=rate)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, rate=rate)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.intermediate = []
        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, rate=1.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, rate=rate))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rate))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)
        # x = self.maxpool(x)

        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)

        feat = self.avgpool(out4)
        out = feat.view(feat.size(0), -1)
        out = self.fc(out)
        return out2, out3, out4, feat, out


def resnet20(num_classes=10, rate=1.0):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [3, 3, 3], num_classes, rate=rate)
    return model


def resnet32(num_classes=10, rate=1.0):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [5, 5, 5], num_classes, rate=rate)
    return model


def resnet44(num_classes=10, rate=1.0):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [7, 7, 7], num_classes, rate=rate)
    return model


def resnet56(num_classes=10, rate=1.0):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [9, 9, 9], num_classes, rate=rate)
    return model


def resnet110(num_classes=10, rate=1.0):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, [18, 18, 18], num_classes, rate=rate)
    return model
