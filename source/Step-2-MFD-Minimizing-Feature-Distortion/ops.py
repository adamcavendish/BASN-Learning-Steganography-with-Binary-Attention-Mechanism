'''
Custom Pytorch Operations
'''
# pylint: disable=C0414, E1101, E1102, W0221, W0235, C0111

import torch
import torch.nn as nn
import torch.autograd

import numpy as np

__all__ = ['op_floor', 'op_ceil', 'op_round', 'lshift', 'rshift', 'gaussian_kernel']


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


class FloorFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inputs):
    return inputs.floor()

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


class Floor(nn.Module):
  def __init__(self):
    super(Floor, self).__init__()

  def forward(self, inputs):
    return FloorFunction.apply(inputs)


op_floor = FloorFunction.apply


class CeilFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inputs):
    return inputs.ceil()

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


class Ceil(nn.Module):
  def __init__(self):
    super(Ceil, self).__init__()

  def forward(self, inputs):
    return CeilFunction.apply(inputs)


op_ceil = CeilFunction.apply


class RoundFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, inputs):
    return inputs.round()

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output


class Round(nn.Module):
  def __init__(self):
    super(Round, self).__init__()

  def forward(self, inputs):
    return RoundFunction.apply(inputs)


op_round = RoundFunction.apply


def lshift(inputs, num_bits):
  if not torch.is_tensor(num_bits):
    num_bits = torch.tensor(num_bits, dtype=torch.float32)
  num_bits = op_floor(num_bits)

  inputs = op_round(inputs)
  inputs = inputs * op_pow_2(num_bits)
  return inputs


def rshift(inputs, num_bits):
  if not torch.is_tensor(num_bits):
    num_bits = torch.tensor(num_bits, dtype=torch.float32)
  num_bits = op_floor(num_bits)

  inputs = op_round(inputs)
  inputs = inputs / op_pow_2(num_bits)
  inputs = op_floor(inputs)
  return inputs
