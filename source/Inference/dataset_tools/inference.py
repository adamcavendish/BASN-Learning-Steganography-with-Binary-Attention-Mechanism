'''
Dataset Loaders for Inference
'''
import pathlib

import PIL.Image

import torch.utils.data as data


class ImageFolderFN(data.Dataset):
  """ImageFolderFN loading an image folder for inference mode

  Returns:
    image data: PIL.Image
    target    : str (class name)
    filename  : str (file's basename)

  Folder Layout Example:
    root/a.png
    root/b.png
    root/dog/xxx.png
    root/dog/yyy.png
    root/cat/aaa.png
    root/cat/bbb.png

  Return Example:
    (PIL.Image, 'unknown',   'a.png')
    (PIL.Image, 'unknown',   'b.png')
    (PIL.Image,     'dog', 'xxx.png')
    (PIL.Image,     'dog', 'yyy.png')
    (PIL.Image,     'cat', 'aaa.png')
    (PIL.Image,     'cat', 'bbb.png')
  """
  def __init__(self, root, transform=None, target_transform=None):
    root_Path = pathlib.Path(root)

    classes = [(d, d.name) for d in sorted(root_Path.iterdir()) if d.is_dir()]

    samples = [(f, 'unknown', f.name) for f in sorted(root_Path.iterdir()) if f.is_file()]
    for class_, class_name in classes:
      samples += [(f, class_name, f.name) for f in sorted(class_.iterdir()) if f.is_file()]

    self.root = root
    self.classes = classes
    self.samples = samples

    self.transform = transform
    self.target_transform = target_transform

  @staticmethod
  def pil_loader(path):
    """PIL Loader for loading image files"""
    with open(path, 'rb') as f:
      img = PIL.Image.open(f)
      return img.convert('RGB')

  def __getitem__(self, index):
    path, target, filename = self.samples[index]
    sample = ImageFolderFN.pil_loader(str(path))

    if not self.transform is None:
      sample = self.transform(sample)

    if not self.target_transform is None:
      target = self.target_transform(target)

    return sample, target, filename

  def __len__(self):
    return len(self.samples)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp,
                                 self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp,
                               self.target_transform.__repr__().replace(
                                   '\n', '\n' + ' ' * len(tmp)))
    return fmt_str
