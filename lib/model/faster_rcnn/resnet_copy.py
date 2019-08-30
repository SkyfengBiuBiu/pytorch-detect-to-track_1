from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.rfcn_copy import _RFCN
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1
  block='basic'
  
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
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4
  block='bottle'
  
  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, dilate_first_conv=False):
    super(Bottleneck, self).__init__()
    if dilate_first_conv:
      self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False, dilation=1)
    else:
      self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    if dilation > 1:
      self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                             padding=dilation, bias=False, dilation=dilation)
    else:
      self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                             padding=1, bias=False, dilation=1)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=1)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

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

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
     
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    
    if block.block == 'basic':
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # a trous 
        #self.layer4 = self._make_dilation_layer(block, 512, layers[3], stride=1, dilation=2)
        # it is slightly better whereas slower to set stride = 1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        print("resnet18 is building")
    elif block.block == 'bottle':
        self.layer1 = self._make_layer1(block, 64, layers[0])
        self.layer2 = self._make_layer1(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer1(block, 256, layers[2], stride=2)
        # a trous 
        self.layer4 = self._make_dilation_layer(block, 512, layers[3], stride=1, dilation=2)
        # it is slightly better whereas slower to set stride = 1
       
        
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


  def _make_dilation_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
	downsample = nn.Sequential(
	    nn.Conv2d(self.inplanes, planes * block.expansion,
		      kernel_size=1, stride=stride, bias=False),
	    nn.BatchNorm2d(planes * block.expansion),
	)


    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
	layers.append(block(self.inplanes, planes, dilation=dilation, dilate_first_conv=True))

    return nn.Sequential(*layers)

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    
    # using resnet18,delete dilation
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  def _make_layer1(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    
    # using resnet18,delete dilation
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes,dilation=dilation))
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
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_RFCN):
  def __init__(self, classes, num_layers=101, pretrained=False, pretrained_rfcn=False, class_agnostic=False):
    
    self.dout_base_model = 512 
    
    if num_layers==101:
        self.model_path = 'data/pretrained_model/res101.pth'
        self.in_features=2048
    elif num_layers==50:
        self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.in_features=2048
    elif num_layers==18:
        self.model_path = 'data/pretrained_model/resnet18-5c106cde.pth'
        self.in_features=512

        
    self.model_rfcn_path = 'data/pretrained_model/rfcn_detect.pth' # trained on Imagenet VID+DET
    
    self.pretrained = pretrained
    self.pretrained_rfcn = pretrained_rfcn
    self.class_agnostic = class_agnostic
    self.num_layers=num_layers


    _RFCN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    if self.num_layers==101:
        resnet = resnet101()
    elif self.num_layers==50:
        resnet = resnet50()
    elif self.num_layers==18:
        resnet = resnet18()    
        
    if self.pretrained == True:
      # Load resnet backbone weights
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RFCN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3, resnet.layer4)

    #self.RCNN_top = nn.Sequential(resnet.layer4)

    #self.RFCN_cls_score = nn.Linear(2048, self.n_classes)
    #if self.class_agnostic:
    #  self.RFCN_bbox_pred = nn.Linear(2048, 4)
    #else:
    #  self.RFCN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RFCN_base[0].parameters(): p.requires_grad=False
    for p in self.RFCN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RFCN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RFCN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
    #In this place, the resnet.layer1 is fixed
      for p in self.RFCN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RFCN_base.apply(set_bn_fix)
    #self.RCNN_top.apply(set_bn_fix)
    # Position-sensitive score maps
    # As in paper, use 3x3 filter with dilation=6 on last res5 feature map instead of 1x1 without dilation
    # With resnet18, I change 2048 to 512

        
    self.RFCN_net = nn.Conv2d(self.in_features, 512, kernel_size=3, padding=6, stride=1, dilation=6)
    self.RFCN_base.add_module("RFCN_net", self.RFCN_net)
    self.RFCN_base.add_module("resnet", resnet.relu)
    nn.init.kaiming_normal(self.RFCN_net.weight.data)

    if self.pretrained_rfcn == True:
      print("Loading pretrained weights from %s" % (self.model_rfcn_path))
      pretrained_rfcn_dict = torch.load(self.model_rfcn_path)['model']
      pretrained_rfcn_dict = {k: v for k, v in pretrained_rfcn_dict.items() if k in self.state_dict()}
      self.state_dict().update(pretrained_rfcn_dict)
      self.load_state_dict(pretrained_rfcn_dict)
    
    self.corr_bbox_net = nn.Conv2d(1051, 4*self.n_reg_classes*7*7, [1,1], padding=0, stride=1)
    nn.init.normal(self.corr_bbox_net.weight.data, 0.0, 0.01)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RFCN_base.eval()
      self.RFCN_base[5].train()
      self.RFCN_base[6].train()
      self.RFCN_base[7].train()
      self.RFCN_base[8].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RFCN_base.apply(set_bn_eval)
   #   self.RFCN_top.apply(set_bn_eval)

  def _im_to_head(self, x):
    conv1 = self.RFCN_base[0](x)
    conv1 = self.RFCN_base[1](conv1)
    conv1 = self.RFCN_base[2](conv1)
    conv1 = self.RFCN_base[3](conv1)
    conv2 = self.RFCN_base[4](conv1)
    conv3 = self.RFCN_base[5](conv2)
    conv4 = self.RFCN_base[6](conv3)
    conv5 = self.RFCN_base[7](conv4)
    top_feat = self.RFCN_base[8](conv5)
    top_feat = self.RFCN_base[9](top_feat)
    return conv3, conv4, conv5, top_feat

#  def _head_to_tail(self, pool5):
#    fc7 = self.RFCN_top(pool5).mean(3).mean(2)
#    return fc7
