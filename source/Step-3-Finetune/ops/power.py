# pylint: disable=no-member

import numpy as np

import torch
import torch.nn as nn


class PowerOfTwoFunction(torch.autograd.Function):
  '''Calculate 2**x'''

  @staticmethod
  def forward(ctx, inputs):
    fwd = torch.pow(2, inputs)
    ctx.save_for_backward(fwd)
    return fwd

  @staticmethod
  def backward(ctx, grad_output):
    fwd, = ctx.saved_tensors
    return fwd * np.log(2) * grad_output


class PowerOfTwo(nn.Module):
  '''Calculate 2**x'''

  def __init__(self):
    super(PowerOfTwo, self).__init__()

  def forward(self, inputs):
    return PowerOfTwoFunction.apply(inputs)


op_pow_2 = PowerOfTwoFunction.apply
