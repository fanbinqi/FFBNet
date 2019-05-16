import sys

import os
import torch
import torch.nn as nn
from utils.timer import Timer
sys.path.append('./')
from models.mobilenet import mobilenet_1
import time
from utils.timer import Timer

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x

class FSSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, size, head, ft_module, pyramid_ext, num_classes):
        super(FSSD, self).__init__()
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.size = size

        # SSD network
        self.base = mobilenet_1()
        # Layer learns to scale the l2 normalized features from conv4_3
        self.ft_module = nn.ModuleList(ft_module)
        self.pyramid_ext = nn.ModuleList(pyramid_ext)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        #self.fea_bn = nn.BatchNorm2d(256, affine=True)
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ft_module), affine=True)
        self.softmax = nn.Softmax()

        self.conv_cat0 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.upsample0 = nn.Upsample(size=(3, 3), mode='bilinear')

        self.conv_cat1 = nn.Conv2d(384, 256, kernel_size=1, padding=0, stride=1)
        self.upsample1 = nn.Upsample(size=(5, 5), mode='bilinear')

        self.conv_cat2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.upsample2 = nn.Upsample(size=(10, 10), mode='bilinear')

        self.conv_cat3 = nn.Conv2d(768, 512, kernel_size=1, padding=0, stride=1)
        self.upsample3 = nn.Upsample(size=(19, 19), mode='bilinear')

        self.conv_cat4 = nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1)
        self.upsample4 = nn.Upsample(size=(38, 38), mode='bilinear')


        self.time = time
        self.timer = Timer

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        source_features = list()
        transformed_features = list()
        loc = list()
        conf = list()

        base_out = self.base(x)

        source_features.append(base_out[0])  # mobilenet 4_1
        source_features.append(base_out[1])  # mobilent_5_5
        source_features.append(base_out[2])  # mobilenet 6_1

        assert len(self.ft_module) == len(source_features)
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(source_features[k]))
        concat_fea = torch.cat(transformed_features, 1)
        x = self.fea_bn(concat_fea)
        fea_bn = x

        # the six detect layers
        pyramid_fea = list()
        for k, v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x)


        #----------this block is to downsample the 1*1 layer to 3*3, and concat with the original 3*3 layer, like Dense connection
        fpn_0 = list()
        detect_5 = pyramid_fea[5]
        detect_4 = pyramid_fea[4]
        detect_5_4 = self.upsample0(detect_5)
        fpn_0.append(detect_4)
        fpn_0.append(detect_5_4)
        detect_4 = torch.cat(fpn_0, 1)
        detect_4 = self.conv_cat0(detect_4)
        pyramid_fea[4] = detect_4
        pyramid_fea[5] = detect_5

        #----------this block is to downsample the 3*3 layer to 5*5, and concat with the original 5*5 layer, like Dense connection
        fpn_1 = list()
        detect_3 = pyramid_fea[3]
        detect_4_3 = self.upsample1(detect_4)
        fpn_1.append(detect_3)
        fpn_1.append(detect_4_3)
        detect_3 = torch.cat(fpn_1, 1)
        detect_3 = self.conv_cat1(detect_3)
        pyramid_fea[3] = detect_3


        #----------this block is to downsample the 5*5 layer to 10*10, and concat with the original 10*10 layer, like Dense connection
        fpn_2 = list()
        detect_2 = pyramid_fea[2]
        detect_3_2 = self.upsample2(detect_3)
        fpn_2.append(detect_2)
        fpn_2.append(detect_3_2)
        detect_2 = torch.cat(fpn_2, 1)
        detect_2 = self.conv_cat2(detect_2)
        pyramid_fea[2] = detect_2


        #----------this block is to downsample the 10*10 layer to 19*19, and concat with the original 19*19 layer, like Dense connection
        fpn_3 = list()
        detect_1 = pyramid_fea[1]
        detect_2_1 = self.upsample3(detect_2)
        fpn_3.append(detect_1)
        fpn_3.append(detect_2_1)
        detect_1 = torch.cat(fpn_3, 1)
        detect_1 = self.conv_cat3(detect_1)
        pyramid_fea[1] = detect_1


        #----------this block is to downsample the 19*19 layer to 38*38, and concat with the original 38*38 layer, like Dense connection
        fpn_4 = list()
        detect_0 = pyramid_fea[0]
        detect_1_0 = self.upsample4(detect_1)
        fpn_4.append(detect_0)
        fpn_4.append(detect_1_0)
        detect_0 = torch.cat(fpn_4, 1)
        detect_0 = self.conv_cat4(detect_0)
        pyramid_fea[0] = detect_0


        # apply multibox head to source layers
        for (x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        #every detect layer's cls and reg
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
            features = ()
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
            features = (
                fea_bn
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            state_dict = torch.load(base_file, map_location=lambda storage, loc: storage)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.base.load_state_dict(new_state_dict)
            print('Finished!')

        else:
            print('Sorry only .pth and .pkl files supported.')

from models.smooth_scale_transfer import *

def feature_transform_module(scale_factor):
    layers = []
    # conv4_1
    layers += [BasicConv(int(256 * scale_factor), 256, kernel_size=1, padding=0)]
    #layers += [down_sample(int(256 * scale_factor), 256)]
    # conv5_5
    layers += [BasicConv(int(512 * scale_factor), 256, kernel_size=1, padding=0, up_size=38)]
    #layers += [BasicConv(int(512 * scale_factor), 256, kernel_size=3, padding=1, stride=2)]
    # conv6_mpo1
    layers += [BasicConv(int(1024 * scale_factor), 256, kernel_size=1, padding=0, up_size=38)]
    #layers += [BasicConv(int(1024 * scale_factor), 256, kernel_size=1, padding=0)]
    return layers



def pyramid_feature_extractor():
    layers = []
    #layers +=  [SST_6(256, 256), SST_5(256, 256), SST_4(256, 256), SST_3(256, 256), SST_2(256, 256), SST_1(256, 256)]
    #
    from models.mobilenet import DepthWiseBlock
    layers = [DepthWiseBlock(256*3, 512, stride=1), DepthWiseBlock(512, 512, stride=2),
              DepthWiseBlock(512, 256, stride=2), DepthWiseBlock(256, 256, stride=2), \
              DepthWiseBlock(256, 128, stride=1, padding=0), DepthWiseBlock(128, 128, stride=1, padding=0)]

    return layers


def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 4, 4],
}
fea_channels = [512, 512, 256, 256, 128, 128]


def build_net(size=512, num_classes=21):
    if size != 300 and size != 512:
        print("Error: Sorry only SSD300 and SSD512 is supported currently!")
        return

    return FSSD(size, multibox(fea_channels, mbox[str(size)], num_classes), feature_transform_module(1),
                pyramid_feature_extractor(), \
                num_classes=num_classes)



#input = torch.tensor(1, 10, 16*10*10).view(1, 16, 10, 10).float()
# pyramid_fea = list()
# for k, v in enumerate(pyramid_feature_extractor()):
#     #x = v(input)
#     pyramid_fea.append(v)
# print(pyramid_fea)


# from torch.autograd import Variable
#
# input1 = Variable(torch.randn(1, 3, 300, 300))
# t = {'im_detect': Timer(), 'misc': Timer()}
# t['im_detect'].tic()
# net = build_net(300,21)
# net = net.forward(input1)
# detect_time = t['im_detect'].toc()
# print(detect_time)
#output = net(input1)
