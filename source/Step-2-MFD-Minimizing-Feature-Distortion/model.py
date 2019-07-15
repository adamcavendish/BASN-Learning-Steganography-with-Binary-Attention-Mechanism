'''
Importance Map Model
'''
# pylint: disable=C0414, E1101, W0221

import operator

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_isinstance(object_, types):
  """Check object_ is an instance of any of types"""
  return reduce(operator.or_, [isinstance(object_, t) for t in types])


def conv3x3(in_planes, out_planes):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def deconv3x3(in_planes, out_planes):
  """3x3 convolution transpose with padding"""
  return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def initialize_module(model, no_init_types=None):
  """Initialize a pytorch Module"""
  for m in model.modules():
    if check_isinstance(m, [nn.Conv2d, nn.ConvTranspose2d]):
      # Convolutions
      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
      # Normalizations
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
    elif check_isinstance(m, [nn.ReLU, nn.ELU, nn.Sigmoid]):
      # Activations
      pass
    elif check_isinstance(m, [type(model), nn.Sequential]):
      # Torch Types
      pass
    elif no_init_types and check_isinstance(m, no_init_types):
      # Customs
      pass
    else:
      raise RuntimeError('Uninitialized layer: %s\n%s' % (type(m), m))


class EncoderBasicBlock(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(EncoderBasicBlock, self).__init__()

    self.conv_fn = conv3x3
    self.norm_fn = lambda: nn.BatchNorm2d(out_planes)
    self.actv_fn = nn.ELU

    self.conv1 = self.conv_fn(in_planes, out_planes)
    self.norm1 = self.norm_fn()
    self.actv1 = self.actv_fn()

    self.conv2 = self.conv_fn(out_planes, out_planes)
    self.norm2 = self.norm_fn()
    self.actv2 = self.actv_fn()

  def forward(self, x):
    residual = x

    x = self.conv1(x)
    x = self.norm1(x)
    x = self.actv1(x)

    x = self.conv2(x)
    x = self.norm2(x)

    x += residual

    x = self.actv2(x)

    return x


class DecoderBasicBlock(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(DecoderBasicBlock, self).__init__()

    self.conv_fn = deconv3x3
    self.norm_fn = lambda: nn.BatchNorm2d(out_planes)
    self.actv_fn = nn.ELU

    self.conv1 = self.conv_fn(in_planes, out_planes)
    self.norm1 = self.norm_fn()
    self.actv1 = self.actv_fn()

    self.conv2 = self.conv_fn(in_planes, out_planes)
    self.norm2 = self.norm_fn()
    self.actv2 = self.actv_fn()

  def forward(self, x):
    residual = x

    x = self.conv1(x)
    x = self.norm1(x)
    x = self.actv1(x)

    x = self.conv2(x)
    x = self.norm2(x)

    x += residual

    x = self.actv2(x)

    return x


class Encoder(nn.Module):
  def __init__(self, block):
    super(Encoder, self).__init__()

    self.conv_fn = nn.Conv2d
    self.norm_fn = nn.BatchNorm2d
    self.actv_fn = nn.ELU

    self.model = nn.Sequential(
        self.conv_fn(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        self.actv_fn(),
        block(64, 64),
        #
        self.conv_fn(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_fn(128),
        self.actv_fn(),
        block(128, 128),
        #
        self.conv_fn(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_fn(256),
        self.actv_fn(),
        block(256, 256),
        #
        self.conv_fn(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_fn(512),
        self.actv_fn(),
    )

    initialize_module(self, no_init_types=[EncoderBasicBlock, DecoderBasicBlock])

  def forward(self, x):
    x = self.model(x)
    return x


class Decoder(nn.Module):
  def __init__(self, block):
    super(Decoder, self).__init__()

    self.conv_fn = nn.ConvTranspose2d
    self.norm_fn = nn.BatchNorm2d
    self.actv_fn = nn.ELU

    self.model = nn.Sequential(
        self.conv_fn(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        self.norm_fn(256),
        self.actv_fn(),
        #
        self.conv_fn(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        self.norm_fn(128),
        self.actv_fn(),
        block(128, 128),
        #
        self.conv_fn(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        self.norm_fn(64),
        self.actv_fn(),
        block(64, 64),
        #
        self.conv_fn(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        self.norm_fn(3),
        self.actv_fn(),
        block(3, 3),
        #
        self.conv_fn(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
    )

    initialize_module(self, no_init_types=[EncoderBasicBlock, DecoderBasicBlock])

  def forward(self, x, epoch=None):
    x = self.model(x)

    if epoch and epoch > 3:
      x = self.actv_fn(x)
      x = F.hardtanh(x, min_val=0.0, max_val=1.0)
    else:
      x = torch.sigmoid(x)

    return x


class Attentioner(nn.Module):
  def __init__(self, encoder_block, decoder_block):
    super(Attentioner, self).__init__()

    self.conv_fn = nn.Conv2d
    self.dcnv_fn = nn.ConvTranspose2d
    self.norm_fn = nn.BatchNorm2d
    self.actv_fn = nn.ELU

    self.encoder_block = encoder_block
    self.decoder_block = decoder_block

    self.model = nn.Sequential(
        self.conv_fn(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
        self.actv_fn(),
        self.encoder_block(64, 64),
        #
        self.conv_fn(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_fn(128),
        self.actv_fn(),
        self.encoder_block(128, 128),
        #
        self.conv_fn(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
        self.norm_fn(256),
        self.actv_fn(),
        self.encoder_block(256, 256),
        #
        self.dcnv_fn(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        self.norm_fn(128),
        self.actv_fn(),
        self.decoder_block(128, 128),
        #
        self.dcnv_fn(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        self.norm_fn(64),
        self.actv_fn(),
        self.decoder_block(64, 64),
        #
        self.dcnv_fn(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.Sigmoid(),
    )

    initialize_module(self, no_init_types=[EncoderBasicBlock, DecoderBasicBlock])

  def forward(self, x):
    return self.model(x)


def encoder():
  return Encoder(EncoderBasicBlock)


def decoder():
  return Decoder(DecoderBasicBlock)


def attentioner():
  return Attentioner(EncoderBasicBlock, DecoderBasicBlock)

