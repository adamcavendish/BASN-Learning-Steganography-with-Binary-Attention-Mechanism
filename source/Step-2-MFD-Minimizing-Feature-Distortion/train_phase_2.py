"""
Train Phase 2: Train Attention Map
"""
# pylint: disable=C0111, C0414, W0603, W0621, E1101, E1102

import json
import time

import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import base_main
import dataset_tools
import model
import ops
import resnet
import utils


class DataObject(object):
  """Misc Data used in Training and Validation"""

  class Metrics(object):
    """All Metrics used in Training and Validation"""

    def __init__(self):
      self.dslt = utils.AverageMetrics()  # DataSet Loading Time
      self.btpt = utils.AverageMetrics()  # BaTch Processing Time

      self.atar = utils.AverageMetrics()  # ATtention ARea

      self.atap = utils.AverageMetrics()  # ATtention Area Penalty
      self.atrl = utils.AverageMetrics()  # ATtention Reconstruction Loss
      self.csrl = utils.AverageMetrics()  # Cover Stego Reconstruction Loss
      self.fmrl = utils.AverageMetrics()  # Feature Map Reconstruction Loss
      self.loss = utils.AverageMetrics()

    def reset_all(self):
      for attr in self.__dict__:
        if isinstance(attr, utils.AverageMetrics):
          attr.reset()

  class Loss(object):
    """All Losses used in Training and Validation"""

    def __init__(self):
      self.atap = None  # ATention Area Penalty
      self.atrl = None  # ATention Reconstruction Loss
      self.csrl = None  # Cover Stego Reconstruction Loss
      self.fmrl = None  # Feature Map Reconstruction Loss
      self.loss = None

  class LossWeights(object):
    """Adaptive Loss Weights used in training (not used)"""

    def __init__(self):
      self.atap = 1.0
      self.atrl = 1.0
      self.csrl = 1.0
      self.fmrl = 1.0

  def __init__(self):
    self.mt = DataObject.Metrics()
    self.ls = DataObject.Loss()

    self.cover_images = None
    self.cover_attentions = None

    self.stego_images = None
    self.stego_attentions = None


class Main(base_main.BaseMain):
  """Main Module"""

  def __init__(self, default_config='config-phase-2.yaml'):
    super(Main, self).__init__(default_config)

    self.dataset_train, self.dataset_valid = self.prepare_datasets()

    # training info
    self.epoch = 0
    self.gidx = 0

    # models
    self.encoder = None
    self.decoder = None
    self.resnet = None
    self.attentioner = None

    # logging info
    self.train = DataObject()
    self.valid = DataObject()

    # training
    self.loss_weights = DataObject.LossWeights()
    self.optm = None

    # Metrics
    self.best_loss = None

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
    # Encoder
    abspath = str(self.pretrain_path / self.config['checkpoint_encoder'])
    self.logger.info('Load encoder from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.encoder.load_state_dict(checkpoint)

    # ResNet
    abspath = str(self.pretrain_path / self.config['checkpoint_resnet'])
    self.logger.info('Load ResNet from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)

    state_dict = self.resnet.state_dict()
    checkpoint = {k: v for k, v in checkpoint.items() if k in state_dict}

    self.resnet.load_state_dict(checkpoint)

    self.logger.info('Pretrained checkpoints loaded.')

  def load_checkpoint(self):
    # Attention Map
    abspath = str(self.checkpoint_path / self.config['checkpoint_attentioner'])
    self.logger.info('Load attentioner from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.attentioner.load_state_dict(checkpoint)

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

    # Attention Map
    abspath = str(pathbase / self.config['checkpoint_attentioner'])
    state_dict = self.attentioner.state_dict()
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

    # Load Pretrained Models
    self.encoder = model.encoder().cuda()
    self.resnet = resnet.resnet18().cuda()

    # Create Models
    self.attentioner = model.attentioner().cuda()

    self.logger.debug('Attentioner Architecture')
    summary(self.attentioner, (3, 224, 224), batch_size=batch_size)

    # Load Pretrained Models
    self.load_pretrained_models()

    model_params = []
    model_params += self.attentioner.parameters()

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

  def run_once(self, mode, noise_mode):
    assert mode in ['train', 'valid']

    train_mode = (mode == 'train')
    do = self.train if mode == 'train' else self.valid

    # forward
    do.cover_attentions = self.attentioner(do.cover_images)

    cover_attentions_mask = ops.op_round(do.cover_attentions * 8)
    do.stego_images = Main.embed_noise(
        do.cover_images, cover_attentions_mask, noise_mode, train_mode=train_mode)

    do.stego_attentions = self.attentioner(do.stego_images)

    # Loss
    images = do.cover_images
    images = utils.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    cover_feature_map = self.resnet(images)

    images = do.stego_images
    images = utils.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    stego_feature_map = self.resnet(images)

    do.ls.fmrl = ((cover_feature_map - stego_feature_map)**2).mean()
    do.ls.csrl = (torch.abs(do.cover_images - do.stego_images)).mean()
    do.ls.atrl = (torch.abs(do.cover_attentions - do.stego_attentions)).mean()

    attention_area = do.cover_attentions.mean()

    # attention area ranges in [0, 1]
    # Punish small attentions since it is most secure not to embed anything.
    do.ls.atap = 0.5 * (1.1 * attention_area)**(8*attention_area-0.1)

    if mode == 'train':
      lw = self.loss_weights
      do.ls.loss = (             \
            lw.fmrl * do.ls.fmrl \
          + lw.csrl * do.ls.csrl \
          + lw.atrl * do.ls.atrl \
          + lw.atap * do.ls.atap )
    else:
      do.ls.loss = do.ls.fmrl + do.ls.csrl + do.ls.atrl + do.ls.atap

    # update metrics
    batch_size = do.cover_images.size(0)
    do.mt.atar.update(attention_area.item(), batch_size)
    do.mt.atap.update(do.ls.atap.item(), batch_size)
    do.mt.atrl.update(do.ls.atrl.item(), batch_size)
    do.mt.csrl.update(do.ls.csrl.item(), batch_size)
    do.mt.fmrl.update(do.ls.fmrl.item(), batch_size)
    do.mt.loss.update(do.ls.loss.item(), batch_size)

  def logging_once(self, mode, noise_mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    if mode == 'train':
      step_info = '[%08d/%08d]' % (step, len(self.dataset_train))
    else:
      step_info = '[%08s/%08d]' % ('-' * 8, len(self.dataset_valid))

    # Print console info
    self.logger.debug(
        '[cover_image_range=(%6.3f, %6.3f)]'
        '[stego_image_range=(%6.3f, %6.3f)]'
        '[loss_weights=(fmrl: %5.0e, csrl: %5.0e, atrl: %5.0e, atap: %5.0e)]',  #
        torch.min(do.cover_images),
        torch.max(do.cover_images),
        torch.min(do.stego_images),
        torch.max(do.stego_images),
        self.loss_weights.fmrl,
        self.loss_weights.csrl,
        self.loss_weights.atrl,
        self.loss_weights.atap)

    self.logger.info(
        '[epoch=%04d/%s]'
        '%s'
        '[noise-mode=%6s]'
        '[time=%6.3f][dslt=%6.3f],'
        '[atar=%6.4f],'
        '[fmrl=%8.4f]'
        '[csrl=%8.4f]'
        '[atrl=%8.4f]'
        '[atap=%8.4f]'
        '[loss=%8.4f]',  #
        self.epoch,
        mode,
        step_info,
        noise_mode,
        do.mt.btpt.avg,
        do.mt.dslt.avg,
        do.mt.atar.avg,
        do.mt.fmrl.avg,
        do.mt.csrl.avg,
        do.mt.atrl.avg,
        do.mt.atap.avg,
        do.mt.loss.avg)

  def image_saving(self, mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    torchvision.utils.save_image(
        do.cover_images.cpu(),  #
        str(self.images_path / f'{mode}-cover-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.stego_images.cpu(),  #
        str(self.images_path / f'{mode}-stego-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_attentions.cpu(),  #
        str(self.images_path / f'{mode}-cover-attn-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.stego_attentions.cpu(),  #
        str(self.images_path / f'{mode}-stego-attn-{self.epoch:04d}-{step:08d}.jpg'))

  def image_saving_detailed(self, mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    name = f'{mode}-cover-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_images.cpu(), path, name)

    name = f'{mode}-stego-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_images.cpu(), path, name)

    name = f'{mode}-cover-attn-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_attentions.cpu(), path, name)

    name = f'{mode}-stego-attn-{self.epoch:04d}-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_attentions.cpu(), path, name)

  def run_train(self):
    logging_interval = self.config['logging_interval']
    heavy_logging_interval = self.config['heavy_logging_interval']

    # switch to train mode
    self.encoder.eval()
    self.resnet.eval()
    self.attentioner.train()

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
          self.image_saving(mode='train', step=step)

        self.gidx += 1

  def run_valid(self):
    noise_mode = 'rand'

    # switch to eval mode
    self.encoder.eval()
    self.resnet.eval()
    self.attentioner.eval()

    with torch.no_grad():
      split_t = time.time()
      for step, (cover_images, _) in enumerate(self.dataset_valid):
        # measure data loading time
        self.valid.mt.dslt.update(time.time() - split_t)

        # data transfer
        self.valid.cover_images = cover_images.cuda()

        # run one batch
        self.run_once(mode='valid', noise_mode=noise_mode)

        # meature time
        self.valid.mt.btpt.update(time.time() - split_t)
        split_t = time.time()

        if step < self.config['valid_images_save_num']:
          self.image_saving(mode='valid', step=step)
          self.image_saving_detailed(mode='valid', step=step)

    self.logging_once(mode='valid', noise_mode=noise_mode, step=None)

    # Save checkpoints
    if self.best_loss is None or self.valid.mt.loss.avg < self.best_loss:
      self.save_checkpoint(
          best_metric_name='loss',
          metric_values={
              'epoch': self.epoch,
              'fmrl': self.valid.mt.fmrl.avg,
              'csrl': self.valid.mt.csrl.avg,
              'atrl': self.valid.mt.atrl.avg,
              'atap': self.valid.mt.atap.avg,
              'loss': self.valid.mt.loss.avg,
          })

    self.save_checkpoint()

    # Update metrics
    if self.best_loss is None:
      self.best_loss = self.valid.mt.loss.avg
    else:
      self.best_loss = min(self.best_loss, self.valid.mt.loss.avg)


if __name__ == '__main__':
  Main().run()
