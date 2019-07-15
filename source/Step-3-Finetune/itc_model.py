'''
Model Definitions
'''

from functools import reduce

import operator

import torch
import torch.nn as nn

import numpy as np

import ops


def check_isinstance(object_, types):
  """Check object_ is an instance of any of types"""
  return reduce(operator.or_, [isinstance(object_, t) for t in types])


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


class Attentioner(nn.Module):
  def __init__(self):
    super(Attentioner, self).__init__()

    self.model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ELU(inplace=True),
        #
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(inplace=True),
        #
        nn.Conv2d(64, 32, 3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ELU(inplace=True),
        #
        nn.Conv2d(32, 3, 3, stride=1, padding=1),
        nn.Sigmoid(),
    )

    initialize_module(self)

  def forward(self, x):
    return self.model(x)


class ImageSmoother(nn.Module):
  def __init__(self, kernel_size=7):
    super(ImageSmoother, self).__init__()
    self.kernel_size = kernel_size

    self.model = nn.Sequential(
        ops.MedianPool2d(kernel_size=self.kernel_size, same=True),
        nn.Hardtanh(min_val=0, max_val=1),
    )

  def forward(self, x):
    return self.model(x)
