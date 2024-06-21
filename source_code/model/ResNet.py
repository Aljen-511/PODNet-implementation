# 这里直接把resnet源码贴进来就可以了
# 但是，需要在每一个阶段(stage)后面加上池化，这里的池化指的是损失函数上的池化
# 下一步是确认输入卷积网络的变量形状，以及卷积网络的输出
# 因为在计算损失的时候，要在每个阶段之后加上池化，所以输出的tensor也需要改变 


'''
Obtained from https://github.com/pytorch/vision/tree/release/0.12
And slightly make some changes to adapt the pratical task
'''
import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}





# 定义并返回特用于bottleneck的几个卷积层
# ------------------以下部分定义Resnet----------------------
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 这里是resnet网络中一个关键的构件，用以减少参数数量，以及保持梯度更新
# 其地位与basicblock同等，18、34用basicblock，50、101、152用bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 完全是pytorch官方的ResNet实现
class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):

        # 这里输入的block即为上面定义的basicblock
        # 这里layers是一个数组，分配对应残差块中卷积层的层数
        # 下面的layer1等命名容易使人误解，实际上它相当于原初resnet的大残差块
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) # 入通道3(三通道图片)，出通道64，卷积核7*7，边缘补齐3行列
        # 注意，这里肯定是
        self.bn1 = nn.BatchNorm2d(64) # 批归一化层
        self.relu = nn.ReLU(inplace=True) # Relu激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 这里遍历所有模块，显式地赋予某些模块初始权值
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 这里遍历所有模块，并将某些模块的初始权重设置为0
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 下面这个if的条件基本一直成立，只在basicblock的layer1不成立
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # 如果要冻结backbone某些构件的更新，可以在这里操作
    def forward(self, x):

        # 输入卷积层的x形状应当是(batch_size, channels, height, width)
        # 具体到此处，形状应当是(batch_size, 3, height, width)
        # h与w在经过卷积层后会改变
        C_1 = self.conv1(x)
        # 这里C_1形状会变成(batch_size, 64, (h-1)/2 + 1, (w-1)/2 + 1)

        C_1 = self.bn1(C_1)
        C_1 = self.relu(C_1)
        # 进行归一化与激活，形状不变

        C_1 = self.maxpool(C_1)
        # 进行最大池化， 形状:(batch_size, h, w) --> (batch_size, 64, (h-1)/2 + 1, (w-1)/2 + 1)

        # 这里的C_3,C_4应该是预留给head的接口，虽然在本项目中只用到了C_5
        # 重头戏来了！！！
        C_2 = self.layer1(C_1)
        C_3 = self.layer2(C_2)
        C_4 = self.layer3(C_3)
        C_5 = self.layer4(C_4)

        # 一波操作猛如虎啊，最后得到的C_5形状是(batch_size, 512*block.expansion, h_unknown, w_unknown)
        # 这里实在不想算最后的h跟w了，还不如直接debug
        # 这里很可能最后的h和w还存在，只是很小，因为此处的resnet对比原来的resnet，少了fc
        # 于是相当于把最后得到的图像特征喂给head
        # 这里不能够只是返回C_5了，应该全都返回，因为最后的损失函数要对这些值做池化操作

        # 同时，根据论文里的说法，这里应该还是有一个fc，但是为什么在具体实现里没找到呢
        # 还有一件事，这里输出的C_5有很多通道，该如何操作呢？


        return [C_1, C_2, C_3, C_4], C_5

#------------------------------------------------
    



def construct_backbone(pretrained = False, nettype = "resnet18"):
    if nettype == "resnet18":
        model = ResNet(BasicBlock,[2,2,2,2])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    elif nettype == "resnet34":
        model = ResNet(BasicBlock, [3,4,6,3])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),strict=False)
    elif nettype == "resnet50":
        model = ResNet(Bottleneck,[3,4,6,3])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),strict=False)
    elif nettype == "resnet101":
        model = ResNet(Bottleneck,[3,4,23,3])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),strict=False)
    elif nettype == "resnet152":
        model = ResNet(Bottleneck,[3,8,36,3])
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']),strict=False)
    return model

# TODO: 修改或者探明ResNet结构，使之能与后面的构件配合起来 -5.31