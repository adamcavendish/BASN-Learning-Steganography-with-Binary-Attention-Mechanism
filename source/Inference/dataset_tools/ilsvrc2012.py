'''
ILSVRC2012 Dataset Loader
'''
import io

import numpy as np
from PIL import Image

import lmdb
import msgpack

import torch.utils.data


class ILSVRC2012(torch.utils.data.Dataset):
  def __init__(self, path, transform=None, target_transform=None):
    self.mdb_path = path

    self.env = lmdb.open(self.mdb_path, readonly=True)
    self.txn = self.env.begin()
    self.entries = self.env.stat()['entries']

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return self.entries

  def __getitem__(self, idx):
    image_rawd = self.txn.get('{:08d}'.format(idx).encode())
    image_info = msgpack.unpackb(image_rawd, encoding='utf-8')
    with Image.open(io.BytesIO(image_info['image'])) as im:
      image = im.convert('RGB')
    target = image_info['label'] - 1  # ILSVRC2012 ID is in range [1, 1000]

    if not self.transform is None:
      image = self.transform(image)

    if not self.target_transform is None:
      target = self.target_transform(target)

    return image, target
