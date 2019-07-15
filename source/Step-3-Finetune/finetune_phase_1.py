"""
FineTune phase-1:
1. ITC Attention Map from Cover Image
2. MFD Attention Map from Cover Image
3. Create Fusion Attention Map
4. Embed Random Noise into Attention Map
5. Finetune ITC Attention Map Loss-es
"""
# pylint: disable=no-member

import difflib
import itertools
import json
import multiprocessing as mp
import time
import typing as tp

import torch
import torch.nn
import torchvision.datasets
import torchvision.transforms as transforms

import PIL.Image
import numpy as np

from torchsummary import summary

import base_main
import dataset_tools
import ops
import utils

import itc_model
import mfd_model


class DataObject(object):
  """Misc Data used in Training and Validation"""

  class Metrics(object):
    """All Metrics used in Training and Validation"""

    def __init__(self):
      self.dslt = utils.AverageMetrics()  # DataSet Loading Time
      self.btpt = utils.AverageMetrics()  # BaTch Processing Time

      self.cs1t = utils.AverageMetrics()  # Cover images Step 1 processing Time
      self.cs2t = utils.AverageMetrics()  # Cover images Step 2 processing Time

      self.ss1t = utils.AverageMetrics()  # Stego images Step 1 processing Time
      self.ss2t = utils.AverageMetrics()  # Stego images Step 2 processing Time

      self.itc_varl = utils.AverageMetrics()  # ITC VARiance Loss
      self.itc_atap = utils.AverageMetrics()  # ITC ATtention Area Penalty
      self.itc_atrl = utils.AverageMetrics()  # ITC ATtention Reconstruction Loss
      self.itc_varr = utils.AverageMetrics()  # ITC VARiance Reduction rate

      self.loss = utils.AverageMetrics()  # Loss

    def reset_all(self):
      """Reset all metrics"""
      for attr in self.__dict__:
        if isinstance(attr, utils.AverageMetrics):
          attr.reset()

  class Loss(object):
    """All Losses used in Training and Validation"""

    def __init__(self):
      self.s1al = None  # Step 1 Attention Loss
      self.s2al = None  # Step 2 Attention Loss
      self.s3al = None  # Step 3 Attention Loss

      self.c1aa = None  # Cover step 1 Attention Area
      self.c2aa = None  # Cover step 2 Attention Area
      self.c3aa = None  # Cover step 3 Attention Area

      self.s1aa = None  # Stego step 1 Attention Area
      self.s2aa = None  # Stego step 2 Attention Area
      self.s3aa = None  # Stego step 3 Attention Area

      self.itc_varl = None  # ITC VARiance Loss
      self.itc_atap = None  # ITC ATtention Area Penalty
      self.itc_atrl = None  # ITC ATtention Reconstruct Loss
      self.itc_varr = None  # ITC VARiance Reduction rate

      self.loss = None

  def __init__(self):
    self.mt = DataObject.Metrics()
    self.ls = DataObject.Loss()

    # Cover Images: torch.Tensor
    self.cover_images = None
    # Cover's ITC Attention Map: torch.Tensor (same shape as cover_images)
    self.cover_itc_attn = None
    # Cover's MFD Attention Map: torch.Tensor (same shape as cover_images)
    self.cover_mfd_attn = None
    # Cover's Fusion Attention Map: torch.Tensor (same shape as cover_images)
    self.cover_fusion_attention = None

    # Stego Images (Use Fusion Attention for Embedding): torch.Tensor (same shape as cover_images)
    self.stego_images = None
    # Stego's ITC Attention Map: torch.Tensor (same shape as cover_images)
    self.stego_itc_attn = None
    # Stego's MFD Attention Map: torch.Tensor (same shape as cover_images)
    self.stego_mfd_attn = None
    # Stego's Fusion Attention Map: torch.Tensor (same shape as cover_images)
    self.stego_fusion_attention = None


class Main(base_main.BaseMain):
  """Main Module"""

  def __init__(self, default_config='config-finetune-1.yaml'):
    super(Main, self).__init__(default_config)

    self.dataset_train, self.dataset_valid = self.prepare_datasets()

    # training info
    self.epoch = 0
    self.gidx = 0

    # models
    self.itc_attentioner = None  # torch.nn.Module
    self.mfd_attentioner = None  # torch.nn.Module

    FusionFunc = tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    self.fusion_algorithms: tp.Dict[FusionFunc] = self.prepare_fusion_algorithms()

    # logging info
    self.train = DataObject()
    self.valid = DataObject()

    # training
    self.image_smoother = None  # torch.nn.Module
    self.variance_pool2d = None  # torch.nn.Module
    self.loss = None
    self.optm = None

    # Metrics
    self.best_loss = None

  def prepare_fusion_algorithms(self):
    """Prepare fusion algorithms"""

    def _min(itc_attens, mfd_attens):
      return torch.min(input=itc_attens, other=mfd_attens)

    def _max(itc_attens, mfd_attens):
      return torch.max(input=itc_attens, other=mfd_attens)

    def _mean(itc_attens, mfd_attens):
      return (itc_attens + mfd_attens) / 2

    return {
        'min': _min,
        'max': _max,
        'mean': _mean,
    }

  def prepare_datasets(self):
    dataset_train = dataset_tools.ILSVRC2012(
        self.config['dataset_train_path'],
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    dataset_valid = dataset_tools.ILSVRC2012(
        self.config['dataset_valid_path'],
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))

    dataset_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=self.config['batch_size'],
        shuffle=True,
        num_workers=self.config['dataset_worker_num'],
        pin_memory=True)
    dataset_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=self.config['batch_size'],
        shuffle=False,
        num_workers=self.config['dataset_worker_num'],
        pin_memory=True)

    return dataset_train, dataset_valid

  def load_pretrained_models(self):
    # ITC Model
    abspath = str(self.pretrain_path / self.config['pretrain_step_1'])
    self.logger.info('Load Step-1 ITC Models from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.itc_attentioner.load_state_dict(checkpoint)

    # MFD Model
    abspath = str(self.pretrain_path / self.config['pretrain_step_2'])
    self.logger.info('Load Step-2 MFD Models from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.mfd_attentioner.load_state_dict(checkpoint)

    self.logger.info('Pretrained checkpoints loaded.')

  def load_checkpoint(self):
    # Finetuned ITC Model
    abspath = str(self.checkpoint_path / self.config['checkpoint_ft_itc'])
    self.logger.info('Load attentioner from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.itc_attentioner.load_state_dict(checkpoint)

    # Optimizer
    abspath = str(self.checkpoint_path / self.config['checkpoint_optm'])
    self.logger.info('Load optimizer from checkpoint: %s', abspath)
    state_dict = torch.load(abspath)

    self.epoch = state_dict.get('epoch', 0) + 1
    checkpoint = state_dict['optimizer']
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.optm.load_state_dict(checkpoint)

    self.logger.info('[epoch=%d] All checkpoints loaded.', self.epoch)

  def save_checkpoint(self, best_metric_name=None, metric_values=None):
    pathbase = self.checkpoint_path
    if best_metric_name:
      pathbase = pathbase / best_metric_name
      pathbase.mkdir(parents=True, exist_ok=True)

    # Finetuned ITC Model
    abspath = str(pathbase / self.config['checkpoint_ft_itc'])
    state_dict = self.itc_attentioner.state_dict()
    torch.save(state_dict, abspath)

    # Optimizer
    abspath = str(pathbase / self.config['checkpoint_optm'])
    state_dict = {
        'epoch': self.epoch,
        'optimizer': self.optm.state_dict(),
    }
    torch.save(state_dict, abspath)

    if metric_values:
      with (pathbase / 'metric_values.json').open('w') as f:
        json.dump(metric_values, f)

  @staticmethod
  def embed_noise(images, lsb_mask, noise_mode, train_mode=True):
    noise_fn_dct = {
        'rand': lambda images: torch.rand_like(images) * 255,
        'ones': lambda images: torch.ones_like(images) * 255,
        'zeros': torch.zeros_like,
    }
    noise_fn = noise_fn_dct[noise_mode]

    embedded_images = images * 255

    if not train_mode:
      embedded_images = torch.clamp(embedded_images, min=0, max=255)

    embedded_images = ops.rshift(embedded_images, lsb_mask)
    embedded_images = ops.lshift(embedded_images, lsb_mask)

    if not train_mode:
      embedded_images = torch.clamp(embedded_images, min=0, max=255)

    with torch.no_grad():
      # Generate random noise to embed, which requires no gradient
      noise = noise_fn(images)
      # Use HSB of the noise
      hsb_mask = 8 - lsb_mask
      noise = ops.rshift(noise, hsb_mask)

    embedded_images = embedded_images + noise
    embedded_images = embedded_images / 255.

    return embedded_images

  def run(self):
    batch_size = self.config['batch_size']
    learning_rate = self.config['learning_rate']

    # Create Models
    self.itc_attentioner = itc_model.Attentioner().cuda()
    self.mfd_attentioner = mfd_model.attentioner().cuda()

    image_smoother = itc_model.ImageSmoother(kernel_size=self.config['smoother_kernel'])
    image_smoother = image_smoother.cuda()
    self.image_smoother = image_smoother

    variance_pool2d = ops.VariancePool2d(kernel_size=self.config['variance_kernel'], same=True)
    variance_pool2d = variance_pool2d.cuda()
    self.variance_pool2d = variance_pool2d

    # Model Summary
    self.logger.debug('ITC Attentioner Architecture')
    summary(self.itc_attentioner, (3, 224, 224), batch_size=batch_size)

    self.logger.debug('MFD Attentioner Architecture')
    summary(self.mfd_attentioner, (3, 224, 224), batch_size=batch_size)

    # Load Pretrained Models
    self.load_pretrained_models()

    model_params = []
    model_params += self.itc_attentioner.parameters()

    self.optm = torch.optim.Adam(model_params, lr=learning_rate)

    # Restore Model
    if not self.args.restart:
      self.load_checkpoint()

    # Setup Global Train Index
    self.gidx = self.epoch * len(self.dataset_train)

    # Initial Validation
    # self.valid = DataObject()
    # self.run_valid()

    total_epochs = self.config['epochs']
    for _ in range(self.epoch, total_epochs):
      utils.adjust_learning_rate(learning_rate, self.optm, self.epoch)

      self.train = DataObject()
      self.run_train()

      self.valid = DataObject()
      self.run_valid()

      self.epoch += 1

  def run_train(self):
    logging_interval = self.config['logging_interval']
    heavy_logging_interval = self.config['heavy_logging_interval']

    # switch to train mode
    self.itc_attentioner.train()
    self.mfd_attentioner.eval()
    self.image_smoother.eval()
    self.variance_pool2d.eval()

    split_t = time.time()
    for step, (cover_images, _) in enumerate(self.dataset_train):
      for noise_mode in ['rand']:
        # measure data loading time
        self.train.mt.dslt.update(time.time() - split_t)

        # data transfer
        self.train.cover_images = cover_images.cuda()

        # run one batch
        self.run_once(mode='train', noise_mode=noise_mode)

        # compute gradients and BP
        self.optm.zero_grad()
        self.train.ls.loss.backward()
        self.optm.step()

        # meature time
        self.train.mt.btpt.update(time.time() - split_t)
        split_t = time.time()

        if step % logging_interval == 0:
          self.logging_once(mode='train', noise_mode=noise_mode, step=step)

          # Reset all metrics
          self.train.mt.reset_all()

        if step % heavy_logging_interval == 0:
          self.image_saving(mode='train', noise_mode=noise_mode, step=step)

        self.gidx += 1

  def run_valid(self):
    # switch to eval mode
    self.itc_attentioner.eval()
    self.mfd_attentioner.eval()
    self.image_smoother.eval()
    self.variance_pool2d.eval()

    split_t = time.time()
    with torch.no_grad():
      for step, (cover_images, _) in enumerate(self.dataset_valid):
        noise_mode = 'rand'

        # measure data loading time
        self.valid.mt.dslt.update(time.time() - split_t)

        # data transfer
        self.valid.cover_images = cover_images.cuda()

        # run one batch
        self.run_once(mode='valid', noise_mode=noise_mode)

        # meature batch processing time
        self.valid.mt.btpt.update(time.time() - split_t)
        split_t = time.time()

        # Save Results
        if step < self.config['valid_images_save_num']:
          self.image_saving(mode='valid', noise_mode=noise_mode, step=step)
          self.image_saving_detailed(mode='valid', noise_mode=noise_mode, step=step)

        # @FIXME
        if step > 20:
          break

    # Logging
    self.logging_once(mode='valid', noise_mode=noise_mode, step=step)

    # Save checkpoints
    if self.best_loss is None or self.valid.mt.loss.avg < self.best_loss:
      self.save_checkpoint(
          best_metric_name='loss',
          metric_values={
              'epoch': self.epoch,
              'loss': self.valid.mt.loss.avg,
              'itc_varl': self.valid.mt.itc_varl.avg,
              'itc_atap': self.valid.mt.itc_atap.avg,
              'itc_atrl': self.valid.mt.itc_atrl.avg,
              'itc_varr': self.valid.mt.itc_varr.avg,
          })

    self.save_checkpoint()

    # Update metrics
    if self.best_loss is None:
      self.best_loss = self.valid.mt.loss.avg
    else:
      self.best_loss = min(self.best_loss, self.valid.mt.loss.avg)

  def run_once(self, mode, noise_mode):
    assert mode in ['train', 'valid']

    train_mode = (mode == 'train')
    do = self.train if mode == 'train' else self.valid

    batch_size = do.cover_images.shape[0]

    # Generate cover attentions
    with utils.metrics_update(do.mt.cs1t, batch_size):
      do.cover_itc_attn = self.itc_attentioner(do.cover_images)
    with utils.metrics_update(do.mt.cs2t, batch_size):
      do.cover_mfd_attn = self.mfd_attentioner(do.cover_images)

    # Fuse together cover attentions
    f_fusion = self.fusion_algorithms[self.config['fusion_algorithm']]
    do.cover_fusion_attention = f_fusion(do.cover_itc_attn, do.cover_mfd_attn)

    # Embed noise
    cover_attentions_mask = ops.op_round(do.cover_fusion_attention * 8)
    do.stego_images = Main.embed_noise(
        do.cover_images, cover_attentions_mask, noise_mode, train_mode=train_mode)

    # Generate stego attentions
    with utils.metrics_update(do.mt.ss1t, batch_size):
      do.stego_itc_attn = self.itc_attentioner(do.stego_images)
    # @FIXME Save CUDA Memory
    # with utils.metrics_update(do.mt.ss2t, batch_size):
    #   do.stego_mfd_attn = self.mfd_attentioner(do.stego_images)

    # @FIXME Save CUDA Memory
    # # Fuse together stego attentions
    # f_fusion = self.fusion_algorithms[self.config['fusion_algorithm']]
    # do.stego_fusion_attention = f_fusion(do.stego_itc_attn, do.stego_mfd_attn)

    # Calculate Loss
    do.ls.s1al = torch.abs(
        Main.create_attention_bitnum(do.cover_itc_attn) -
        Main.create_attention_bitnum(do.stego_itc_attn)).sum(dim=1).mean()
    # @FIXME Save CUDA Memory
    # do.ls.s2al = torch.abs(
    #     Main.create_attention_bitnum(do.cover_mfd_attn) -
    #     Main.create_attention_bitnum(do.stego_mfd_attn)).sum(dim=1).mean()
    # do.ls.s3al = torch.abs(
    #     Main.create_attention_bitnum(do.cover_fusion_attention) -
    #     Main.create_attention_bitnum(do.stego_fusion_attention)).sum(dim=1).mean()
    do.ls.s2al = -1
    do.ls.s3al = -1

    do.ls.c1aa = do.cover_itc_attn.mean()
    do.ls.c2aa = do.cover_mfd_attn.mean()
    do.ls.c3aa = do.cover_fusion_attention.mean()

    do.ls.s1aa = do.stego_itc_attn.mean()
    # do.ls.s2aa = do.stego_mfd_attn.mean()
    # do.ls.s3aa = do.stego_fusion_attention.mean()
    do.ls.s2aa = -1
    do.ls.s3aa = -1

    # Calculate Loss
    loss_weights = self.config['loss_weights']
    cover_smthed = self.image_smoother(do.cover_images)
    weighed_cover_smthed =                 \
          do.cover_itc_attn * cover_smthed \
        + (1 - do.cover_itc_attn) * do.cover_images

    weighed_cover_smthed_var = self.variance_pool2d(weighed_cover_smthed)
    cover_images_var = self.variance_pool2d(do.cover_images)
    cover_smthed_var = self.variance_pool2d(cover_smthed)

    do.ls.itc_varl = torch.mean(weighed_cover_smthed_var)

    # area_penalty ranges in [0, 1]. We decay the area penalty to get working attentions.
    # Punish more when closing to 1, less when closing to 0.
    itc_area_penalty = do.ls.c1aa
    itc_area_penalty = itc_area_penalty**(3 - 2 * itc_area_penalty)
    do.ls.itc_atap = itc_area_penalty

    do.ls.itc_atrl = ((do.cover_itc_attn - do.stego_itc_attn)**2).mean()

    do.ls.loss =                          \
          loss_weights[0] * do.ls.itc_varl \
        + loss_weights[1] * do.ls.itc_atap \
        + loss_weights[2] * do.ls.itc_atrl

    # calculate metric helpers
    itc_var_cur = torch.mean(weighed_cover_smthed_var)
    itc_var_min = torch.mean(cover_smthed_var)
    itc_var_max = torch.mean(cover_images_var)
    itc_variance_reduction = (itc_var_max - itc_var_cur) / (itc_var_max - itc_var_min)
    do.ls.itc_varr = itc_variance_reduction

    # update meters
    do.mt.itc_varl.update(do.ls.itc_varl.item(), batch_size)
    do.mt.itc_atap.update(do.ls.itc_atap.item(), batch_size)
    do.mt.itc_atrl.update(do.ls.itc_atrl.item(), batch_size)
    do.mt.itc_varr.update(do.ls.itc_varr.item(), batch_size)
    do.mt.loss.update(do.ls.loss.item(), batch_size)

  @staticmethod
  def create_attention_bitnum(attention):
    return torch.round(attention * 8).view(attention.size(0), -1)

  def image_saving(self, mode, noise_mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    torchvision.utils.save_image(
        do.cover_images.cpu(),  #
        str(self.images_path / f'{mode}-cover-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_itc_attn.cpu(),  #
        str(self.images_path / f'{mode}-cover-itc-attn-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_mfd_attn.cpu(),  #
        str(self.images_path / f'{mode}-cover-mfd-attn-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_fusion_attention.cpu(),  #
        str(self.images_path / f'{mode}-cover-fusion-attention-{self.epoch:04d}-{step:08d}.jpg'))

    torchvision.utils.save_image(
        do.stego_images.cpu(),  #
        str(self.images_path / f'{mode}-stego-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.stego_itc_attn.cpu(),  #
        str(self.images_path / f'{mode}-stego-itc-attn-{self.epoch:04d}-{step:08d}.jpg'))
    # @FIXME Save CUDA Memory
    # torchvision.utils.save_image(
    #     do.stego_mfd_attn.cpu(),  #
    #     str(self.images_path / f'{mode}-stego-mfd-attn-{self.epoch:04d}-{step:08d}.jpg'))
    # torchvision.utils.save_image(
    #     do.stego_fusion_attention.cpu(),  #
    #     str(self.images_path / f'{mode}-stego-fusion-attention-{self.epoch:04d}-{step:08d}.jpg'))

  def image_saving_detailed(self, mode, noise_mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    name = f'{mode}-cover-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_images.cpu(), path, name)

    name = f'{mode}-cover-itc-attn-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_itc_attn.cpu(), path, name)

    name = f'{mode}-cover-mfd-attn-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_mfd_attn.cpu(), path, name)

    name = f'{mode}-cover-fusion-attention-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_fusion_attention.cpu(), path, name)

    name = f'{mode}-stego-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_images.cpu(), path, name)

    name = f'{mode}-stego-itc-attn-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_itc_attn.cpu(), path, name)

    # @FIXME Save CUDA Memory
    # name = f'{mode}-stego-mfd-attn-{self.epoch:04d}-{step:08d}'
    # path = self.images_path / name
    # path.mkdir(parents=True, exist_ok=True)
    # utils.save_image_batch(do.stego_mfd_attn.cpu(), path, name)

    # name = f'{mode}-stego-fusion-attention-{self.epoch:04d}-{step:08d}'
    # path = self.images_path / name
    # path.mkdir(parents=True, exist_ok=True)
    # utils.save_image_batch(do.stego_fusion_attention.cpu(), path, name)

  def logging_once(self, mode, noise_mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    if mode == 'train':
      step_info = '[%08d/%08d]' % (step, len(self.dataset_train))
    else:
      step_info = '[%08s/%08d]' % ('-' * 8, len(self.dataset_valid))

    # Print console info
    self.logger.info(
        '[epoch=%04d/%s]'
        '%s'
        '[dslt=%6.3f]'
        '[btpt=%6.3f],'
        '[cs1t=%6.3f]'
        '[cs2t=%6.3f]'
        '[ss1t=%6.3f]'
        '[ss2t=%6.3f]',  #
        self.epoch,
        mode,
        step_info,
        do.mt.dslt.avg,
        do.mt.btpt.avg,
        do.mt.cs1t.avg,
        do.mt.cs2t.avg,
        do.mt.ss1t.avg,
        do.mt.ss2t.avg)
    self.logger.info(
        '  '
        '[step-1-attn-loss=%8.1fb]'
        '[step-2-attn-loss=%8.1fb]'
        '[step-3-attn-loss=%8.1fb]'
        '[step-1-area=%5.3f/%5.3f]'
        '[step-2-area=%5.3f/%5.3f]'
        '[step-3-area=%5.3f/%5.3f]',  #
        do.ls.s1al,
        do.ls.s2al,
        do.ls.s3al,
        do.ls.c1aa,
        do.ls.s1aa,
        do.ls.c2aa,
        do.ls.s2aa,
        do.ls.c3aa,
        do.ls.s3aa)

    self.logger.info(
        '  '
        '[itc_varl=%6.3f]'
        '[itc_atap=%6.3f]'
        '[itc_atrl=%6.3f]'
        '[    loss=%6.3f]'
        '[itc_varr=%5.2f%%]',  #
        do.mt.itc_varl.avg,
        do.mt.itc_atap.avg,
        do.mt.itc_atrl.avg,
        do.mt.loss.avg,
        do.mt.itc_varr.avg * 100)


if __name__ == '__main__':
  Main().run()
