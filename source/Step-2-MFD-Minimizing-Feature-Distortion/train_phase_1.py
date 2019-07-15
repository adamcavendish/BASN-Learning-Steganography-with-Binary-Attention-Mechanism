"""
Train Phase 1: Train Autoencoder
"""
# pylint: disable=C0111, C0414, W0603, W0621, E1101

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
import utils


class DataObject(object):
  """Misc Data used in Training and Validation"""

  class Metrics(object):
    """All Metrics used in Training and Validation"""

    def __init__(self):
      self.dslt = utils.AverageMetrics()  # DataSet Loading Time
      self.btpt = utils.AverageMetrics()  # BaTch Processing Time

      self.loss = utils.AverageMetrics()

    def reset_all(self):
      for attr in self.__dict__:
        if isinstance(attr, utils.AverageMetrics):
          attr.reset()

  class Loss(object):
    """All Losses used in Training and Validation"""

    def __init__(self):
      self.loss = None

  def __init__(self):
    self.mt = DataObject.Metrics()
    self.ls = DataObject.Loss()

    self.orig_images = None
    self.embeddings = None
    self.rcst_images = None


class Main(base_main.BaseMain):
  """Main Module"""

  def __init__(self, default_config='config-phase-1.yaml'):
    super(Main, self).__init__(default_config)

    self.dataset_train, self.dataset_valid = self.prepare_datasets()

    # training info
    self.epoch = 0
    self.gidx = 0

    # models
    self.encoder = None
    self.decoder = None

    # logging info
    self.train = DataObject()
    self.valid = DataObject()

    # training
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

  def load_checkpoint(self):
    # Encoder
    abspath = str(self.checkpoint_path / self.config['checkpoint_encoder'])
    self.logger.info('Load encoder from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.encoder.load_state_dict(checkpoint)

    # Decoder
    abspath = str(self.checkpoint_path / self.config['checkpoint_decoder'])
    self.logger.info('Load decoder from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.decoder.load_state_dict(checkpoint)

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

    # Encoder
    abspath = str(pathbase / self.config['checkpoint_encoder'])
    state_dict = self.encoder.state_dict()
    torch.save(state_dict, abspath)

    # Decoder
    abspath = str(pathbase / self.config['checkpoint_decoder'])
    state_dict = self.decoder.state_dict()
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

  def run(self):
    batch_size = self.config['batch_size']
    learning_rate = self.config['learning_rate']

    # Create Model
    self.encoder = model.encoder().cuda()
    self.decoder = model.decoder().cuda()

    self.logger.debug('Encoder Architecture')
    summary(self.encoder, (3, 224, 224), batch_size=batch_size)

    self.logger.debug('Decoder Architecture')
    summary(self.decoder, (512, 14, 14), batch_size=batch_size)

    model_params = []
    model_params += self.encoder.parameters()
    model_params += self.decoder.parameters()

    self.optm = torch.optim.SGD(
        model_params,
        lr=learning_rate,
        momentum=self.config['momentum'],
        weight_decay=self.config['weight_decay'])

    # Restore Model
    if not self.args.restart:
      self.load_checkpoint()

    # Setup Global Train Index
    self.gidx = self.epoch * len(self.dataset_train)

    # Initial Validation
    self.valid = DataObject()
    self.run_valid()

    total_epochs = self.config['epochs']
    for _ in range(self.epoch, total_epochs):
      utils.adjust_learning_rate(learning_rate, self.optm, self.epoch)

      self.train = DataObject()
      self.run_train()

      self.valid = DataObject()
      self.run_valid()

      self.epoch += 1

  def run_once(self, mode):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    do.embeddings = self.encoder(do.orig_images)
    do.rcst_images = self.decoder(do.embeddings)
    do.ls.loss = ((do.orig_images - do.rcst_images)**2).mean(dim=0).sum()

    # update meters
    batch_size = do.orig_images.size(0)
    do.mt.loss.update(do.ls.loss.item(), batch_size)

  def logging_once(self, mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    if mode == 'train':
      step_info = '[%08d/%08d]' % (step, len(self.dataset_train))
    else:
      step_info = '[%08s/%08d]' % ('-' * 8, len(self.dataset_valid))

    # Print console info
    self.logger.debug(
        '[orig_image_range=(%6.3f, %6.3f)]'
        '[rcst_image_range=(%6.3f, %6.3f)]',  #
        torch.min(do.orig_images),
        torch.max(do.orig_images),
        torch.min(do.rcst_images),
        torch.max(do.rcst_images))

    self.logger.info(
        '[epoch=%04d/%s]'
        '%s'
        '[time=%6.3f][dslt=%6.3f],'
        '[loss=%8.4f]',  #
        self.epoch,
        mode,
        step_info,
        do.mt.btpt.avg,
        do.mt.dslt.avg,
        do.mt.loss.avg)

  def image_saving(self, mode, step):
    assert mode in ['train', 'valid']

    do = self.train if mode == 'train' else self.valid

    torchvision.utils.save_image(
        do.orig_images.cpu(),  #
        str(self.images_path / f'{mode}-orig-{self.epoch:04d}-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.rcst_images.cpu(),  #
        str(self.images_path / f'{mode}-rcst-{self.epoch:04d}-{step:08d}.jpg'))

  def run_train(self):
    logging_interval = self.config['logging_interval']
    heavy_logging_interval = self.config['heavy_logging_interval']

    # switch to train mode
    self.encoder.train()
    self.decoder.train()

    split_t = time.time()
    for step, (images, _) in enumerate(self.dataset_train):
      # measure data loading time
      self.train.mt.dslt.update(time.time() - split_t)

      # data transfer
      self.train.orig_images = images.cuda()

      # run one batch
      self.run_once(mode='train')

      # compute gradients and BP
      self.optm.zero_grad()
      self.train.ls.loss.backward()
      self.optm.step()

      # meature time
      self.train.mt.btpt.update(time.time() - split_t)
      split_t = time.time()

      if step % logging_interval == 0:
        self.logging_once(mode='train', step=step)

        # Reset all metrics
        self.train.mt.reset_all()

      if step % heavy_logging_interval == 0:
        self.image_saving(mode='train', step=step)

      self.gidx += 1

  def run_valid(self):
    # switch to evaluate mode
    self.encoder.eval()
    self.decoder.eval()

    with torch.no_grad():
      split_t = time.time()
      for step, (images, _) in enumerate(self.dataset_valid):
        # measure data loading time
        self.valid.mt.dslt.update(time.time() - split_t)

        # data transfer
        self.valid.orig_images = images.cuda()

        # run one batch
        self.run_once(mode='valid')

        # meature time
        self.valid.mt.btpt.update(time.time() - split_t)
        split_t = time.time()

        if step < self.config['valid_images_save_num']:
          self.image_saving(mode='valid', step=step)

      self.logging_once(mode='valid', step=None)

    # Save checkpoints
    if self.best_loss is None or self.valid.mt.loss.avg < self.best_loss:
      self.save_checkpoint(
          best_metric_name='loss',
          metric_values={
              'epoch': self.epoch,
              'loss': self.valid.mt.loss.avg
          })

    self.save_checkpoint()

    # Update metrics
    if self.best_loss is None:
      self.best_loss = self.valid.mt.loss.avg
    else:
      self.best_loss = min(self.best_loss, self.valid.mt.loss.avg)


if __name__ == '__main__':
  Main().run()
