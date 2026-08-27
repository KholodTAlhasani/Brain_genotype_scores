
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

__all__ = ['MultiTaskResNet', 'multitask_resnet10', 'multitask_resnet18', 'multitask_resnet34', 'multitask_resnet50']



def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out, zero_pads], dim=1)


    return out


# ---------------------------
# Squeeze-and-Excitation 3D Block
# ---------------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16): 
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, d, h, w = x.size()
        # Squeeze: global average pooling to (b, c)
        y = self.avg_pool(x).view(b, c)
        # Excitation: compute channel weights
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# ---------------------------
# Spatial Attention 3D Module
# ---------------------------
class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        # The convolution takes a 2-channel input (avg and max along channel dim)
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, D, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)     # [B, 2, D, H, W]
        sa = self.conv(x_cat)
        sa = self.sigmoid(sa)
        return x * sa
    
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_se=False, reduction=16, use_sa=False, sa_kernel_size=7):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock3D(planes, reduction=reduction)
        
        self.use_sa = use_sa
        if self.use_sa:
            self.sa = SpatialAttention3D(kernel_size=sa_kernel_size)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE (channel attention) if enabled
        if self.use_se:
            out = self.se(out)
        # Apply Spatial Attention if enabled
        if self.use_sa:
            out = self.sa(out)
            
            
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_se=False, reduction=16, use_sa=False, sa_kernel_size=7):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock3D(planes * 4, reduction=reduction)
            
        self.use_sa = use_sa
        if self.use_sa:
            self.sa = SpatialAttention3D(kernel_size=sa_kernel_size)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_se:
            out = self.se(out)
            
        if self.use_sa:
            out = self.sa(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MultiTaskResNet(nn.Module):
    def __init__(self, block, layers, num_tasks, num_classes_per_task, shortcut_type='B', no_cuda=False, use_se=False, reduction=16, use_sa=False, sa_kernel_size=7):
        super(MultiTaskResNet, self).__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.use_se = use_se
        self.reduction = reduction
        self.use_sa = use_sa
        self.sa_kernel_size = sa_kernel_size
        
        # Initial layers (feature extractor) remain unchanged
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, use_se=self.use_se, reduction=self.reduction, use_sa=self.use_sa, sa_kernel_size=self.sa_kernel_size)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2, use_se=self.use_se, reduction=self.reduction, use_sa=self.use_sa, sa_kernel_size=self.sa_kernel_size)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2, use_se=self.use_se, reduction=self.reduction, use_sa=self.use_sa, sa_kernel_size=self.sa_kernel_size)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4, use_se=self.use_se, reduction=self.reduction, use_sa=self.use_sa, sa_kernel_size=self.sa_kernel_size)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Define multiple classification heads (one per task)
        self.fc_tasks = nn.ModuleList([nn.Linear(512 * block.expansion, num_classes_per_task)
                                       for _ in range(num_tasks)])
        
        # Learnable log sigmas for each task
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

        # Initialize new FC layers (the backbone layers already have pretrained weights)
        for fc in self.fc_tasks:
            nn.init.kaiming_normal_(fc.weight)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1, use_se=False, reduction=16, use_sa=False, sa_kernel_size=7):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion,
                                       stride=stride, no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, use_se=use_se, reduction=reduction, use_sa=use_sa, sa_kernel_size=sa_kernel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, use_se=use_se, reduction=reduction, use_sa=use_sa, sa_kernel_size=sa_kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten

        # Compute outputs for each task
        outputs = [fc(x) for fc in self.fc_tasks]
        return outputs  # List of logits for each task



def multitask_resnet10(num_tasks, num_classes_per_task, shortcut_type, no_cuda, use_se, reduction, use_sa, sa_kernel_size):
    return MultiTaskResNet(BasicBlock, [1, 1, 1, 1], num_tasks, num_classes_per_task, shortcut_type, no_cuda, use_se, reduction, use_sa, sa_kernel_size)

def multitask_resnet18(num_tasks, num_classes_per_task, shortcut_type, no_cuda, use_se, reduction, use_sa, sa_kernel_size):
    return MultiTaskResNet(BasicBlock, [2, 2, 2, 2], num_tasks, num_classes_per_task, shortcut_type, no_cuda, use_se, reduction, use_sa, sa_kernel_size)

def multitask_resnet34(num_tasks=3, num_classes_per_task=3, shortcut_type='B', no_cuda=False):
    return MultiTaskResNet(BasicBlock, [3, 4, 6, 3], num_tasks, num_classes_per_task, shortcut_type, no_cuda)

def multitask_resnet50(num_tasks, num_classes_per_task, shortcut_type, no_cuda, use_se, reduction, use_sa, sa_kernel_size):
    return MultiTaskResNet(Bottleneck, [3, 4, 6, 3], num_tasks, num_classes_per_task, shortcut_type, no_cuda, use_se, reduction, use_sa, sa_kernel_size)

def multitask_resnet101(num_tasks=3, num_classes_per_task=3, shortcut_type='B', no_cuda=False):
    return MultiTaskResNet(Bottleneck, [3, 4, 23, 3], num_tasks, num_classes_per_task, shortcut_type, no_cuda)

def multitask_resnet152(num_tasks=3, num_classes_per_task=3, shortcut_type='B', no_cuda=False):
    return MultiTaskResNet(Bottleneck, [3, 8, 36, 3], num_tasks, num_classes_per_task, shortcut_type, no_cuda)

def multitask_resnet200(num_tasks=3, num_classes_per_task=3, shortcut_type='B', no_cuda=False):
    return MultiTaskResNet(Bottleneck, [3, 24, 36, 3], num_tasks, num_classes_per_task, shortcut_type, no_cuda)

