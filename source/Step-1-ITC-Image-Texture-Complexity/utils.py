'''
Utilitiy Functions
'''

import PIL.Image
import collections
import math
import numpy as np
import skimage
import skimage.feature
import torch


class Edges(object):
  def __call__(self, sample):
    image = np.array(sample) / 255.

    soft_min = 0.05
    soft_max = 0.95

    canny_edges_mask = skimage.feature.canny(skimage.color.rgb2gray(image))
    canny_edges = np.zeros_like(canny_edges_mask) + soft_min
    canny_edges[canny_edges_mask] = soft_max

    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    canny_edges = torch.tensor(np.expand_dims(canny_edges, 0), dtype=torch.float32)

    return image, canny_edges


def adjust_learning_rate(learning_rate, optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = learning_rate * (0.1**(epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(predictions, labels, top_k=(1, )):
  """Computes the top-k(s) for the specified values of k"""
  with torch.no_grad():
    max_k = max(top_k)
    batch_size = labels.size(0)

    _, pred = predictions.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    def _get_accuracy_from_correct(k):
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      return correct_k.mul_(100.0 / batch_size)

    return [_get_accuracy_from_correct(k) for k in top_k]


def remove_data_parallel_prefix(state_dict):
  """Remove DataParallel Prefix for Model Restoring"""
  new_state_dict = collections.OrderedDict()
  for k, v in state_dict.items():
    if k.startswith('module.'):
      k = k[7:]
    new_state_dict[k] = v
  return new_state_dict


def format_topk_info(meta, prob, topk, classname, filename):
  """format topk information"""
  topk_info = []
  for prob_i, topk_i in zip(prob, topk):
    words = meta[topk_i][b'words'].decode()
    topk_info.append('%s[%04d](%4.2f%%)' % (words, topk_i, prob_i * 100))
  topk_info = ', '.join(topk_info)
  return '%s(%s): %s' % (filename, classname, topk_info)


def image_grid(images, padding=3, pad_value=0):
  '''
  Images:
      numpy.ndarray, NHWC type
  '''
  batch_size = images.shape[0]
  nrows, ncols, ncnls = images.shape[1:]

  impcol = math.ceil(math.sqrt(batch_size))
  improw = math.ceil(batch_size / impcol)

  ret_rows = improw * nrows + padding * (improw + 1)
  ret_cols = impcol * ncols + padding * (impcol + 1)

  ret = np.ones((ret_rows, ret_cols, ncnls)) * pad_value

  for ridx in range(improw):
    for cidx in range(impcol):
      idx = ridx * impcol + cidx
      if idx >= batch_size:
        break
      img = images[idx]
      rlb = (padding + nrows) * ridx + padding
      rub = (padding + nrows) * (ridx + 1)
      clb = (padding + ncols) * cidx + padding
      cub = (padding + ncols) * (cidx + 1)
      ret[rlb:rub, clb:cub, :] = img

  return ret


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    """Reset current meter"""
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    """Update current meter with val (n is the batch size)"""
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class DeNormalize(object):
  """De-Normalize a tensor image with mean and standard deviation.
  Inverse operation to torchvision.transforms.Normalize
  """

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor) and tensor.ndimension() == 3:
      raise TypeError('tensor is not a torch image.')

    tensor = tensor.clone().detach()

    # This is faster than using broadcasting, don't change without benchmarking
    for t, m, s in zip(tensor, self.mean, self.std):
      t.mul_(s).add_(m)

    return tensor

  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
