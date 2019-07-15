# pylint: disable=not-callable, no-member

from . import roundoff
from . import power

import torch


def lshift(inputs, num_bits):
  if not torch.is_tensor(num_bits):
    num_bits = torch.tensor(num_bits, dtype=torch.float32)
  num_bits = roundoff.op_floor(num_bits)

  inputs = roundoff.op_round(inputs)
  inputs = inputs * power.op_pow_2(num_bits)
  return inputs


def rshift(inputs, num_bits):
  if not torch.is_tensor(num_bits):
    num_bits = torch.tensor(num_bits, dtype=torch.float32)
  num_bits = roundoff.op_floor(num_bits)

  inputs = roundoff.op_round(inputs)
  inputs = inputs / power.op_pow_2(num_bits)
  inputs = roundoff.op_floor(inputs)
  return inputs
