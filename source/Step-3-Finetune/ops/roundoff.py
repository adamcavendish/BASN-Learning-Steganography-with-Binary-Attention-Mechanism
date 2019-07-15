import torch
import torch.nn as nn


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
