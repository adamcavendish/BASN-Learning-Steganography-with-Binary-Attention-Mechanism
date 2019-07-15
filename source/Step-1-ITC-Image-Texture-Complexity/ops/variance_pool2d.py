'''
Variance Pooling 2D
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class VariancePool2d(nn.Module):
  """ Variance Pool 2D module.

  Args:
    kernel_size: size of pooling kernel, int or 2-tuple
    stride: pool stride, int or 2-tuple
    padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
    same: override padding and enforce same padding, boolean
  """

  def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
    super(VariancePool2d, self).__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.same = same

    self._k = _pair(kernel_size)
    self._s = _pair(stride)
    self._p = _quadruple(padding)  # convert to l, r, t, b

    self.pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride)

  def _padding(self, x):
    if self.same:
      ih, iw = x.size()[2:]
      if ih % self._s[0] == 0:
        ph = max(self._k[0] - self._s[0], 0)
      else:
        ph = max(self._k[0] - (ih % self._s[0]), 0)
      if iw % self._s[1] == 0:
        pw = max(self._k[1] - self._s[1], 0)
      else:
        pw = max(self._k[1] - (iw % self._s[1]), 0)
      pl = pw // 2
      pr = pw - pl
      pt = ph // 2
      pb = ph - pt
      padding = (pl, pr, pt, pb)
    else:
      padding = self._p
    return padding

  def forward(self, x):
    x = F.pad(x, self._padding(x), mode='reflect')
    x = self.pool(x * x) - self.pool(x)**2
    return x
