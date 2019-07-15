'''
Train Phase 1: Train Embedding Attention
'''
# pylint: disable=C0111, C0414, W0603, W0621, E1101

import argparse
import datetime
import json
import logging
import pathlib
import random
import time

from ruamel.yaml import YAML
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import dataset_tools
import model
import ops
import utils


class Main(object):
  """Main Module"""

  def __init__(self):
    self.args, self.config = Main.prepare_cmd_args()
    self.unique_id = self.gen_unique_id()

    self.logging_path = None
    self.images_path = None
    self.checkpoint_path = None
    self.prepare_directories()

    self.logger = self.prepare_loggers()

    self.prepare_seed_and_cudnn()

    self.dataset_train, self.dataset_valid = self.prepare_datasets()

    # Initialized in run()
    self.epoch = 0
    self.gidx = 0

    self.attentioner = None
    self.image_smoother = None
    self.variance_pool2d = None
    self.loss = None

    self.optm = None

    # Metrics
    self.best_loss = None

  @staticmethod
  def prepare_cmd_args():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config-phase-1.yaml', help='configuration file path')
    parser.add_argument('--comment', default='', help='current training comment')
    parser.add_argument('--restart', action='store_true', default=False)
    args = parser.parse_args()

    config_path = args.config

    # Setup configurations
    with open(config_path, 'r') as f:
      yaml = YAML()
      config = yaml.load(f)

    return args, config

  def gen_unique_id(self):
    def _gen_fullname(name, run_id, comment=''):
      if comment:
        return f'{name}-{run_id}-{comment}'
      return f'{name}-{run_id}'

    name = 'phase-%d-%s' % (self.config['task_phase'], self.config['task_name'])
    # integer timestamp last 10 digits as run_id
    run_id = ('%010d' % int(datetime.datetime.now().timestamp()))[-10:]
    comment = self.args.comment
    fullname = _gen_fullname(name, run_id, comment)

    return fullname

  def prepare_directories(self):
    self.logging_path = pathlib.Path(self.config['logging_path'])
    self.logging_path.mkdir(parents=True, exist_ok=True)

    self.images_path = self.logging_path / 'images' / self.unique_id
    self.images_path.mkdir(parents=True, exist_ok=True)

    self.checkpoint_path = pathlib.Path(self.config['checkpoint_path'])
    self.checkpoint_path.mkdir(parents=True, exist_ok=True)

  def prepare_loggers(self):
    logging.basicConfig(
        format='[%(asctime)s][%(levelname)-5.5s] %(message)s',
        handlers=[
            logging.FileHandler(str(self.logging_path / f'{self.unique_id}.log')),
            logging.StreamHandler(),
        ])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger

  def prepare_seed_and_cudnn(self):
    seed = self.config['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if self.config['deterministic']:
      cudnn.deterministic = True

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
    # Attentioner
    abspath = str(self.checkpoint_path / self.config['checkpoint_attentioner'])
    self.logger.info('Load attentioner from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    self.attentioner.load_state_dict(checkpoint)

    # Optimizer
    abspath = str(self.checkpoint_path / self.config['checkpoint_optm'])
    self.logger.info('Load optimizer from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)

    self.epoch = checkpoint.get('epoch', 0) + 1
    self.optm.load_state_dict(checkpoint['optimizer'])

    self.logger.info('[epoch=%d] All checkpoints loaded.', self.epoch)

  def save_checkpoint(self, best_metric_name=None, metric_values=None):
    pathbase = self.checkpoint_path
    if best_metric_name:
      pathbase = pathbase / best_metric_name
      pathbase.mkdir(parents=True, exist_ok=True)

    # Attentioner
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

  def run(self):
    batch_size = self.config['batch_size']
    learning_rate = self.config['learning_rate']

    # Create Model
    attentioner = model.Attentioner()
    attentioner = torch.nn.DataParallel(attentioner).cuda()
    self.attentioner = attentioner

    image_smoother = model.ImageSmoother(kernel_size=self.config['smoother_kernel'])
    image_smoother = torch.nn.DataParallel(image_smoother).cuda()
    self.image_smoother = image_smoother

    variance_pool2d = ops.VariancePool2d(kernel_size=self.config['variance_kernel'], same=True)
    variance_pool2d = torch.nn.DataParallel(variance_pool2d).cuda()
    self.variance_pool2d = variance_pool2d

    self.logger.debug('Attentioner Architecture')
    summary(self.attentioner, (3, 224, 224), batch_size=batch_size)

    model_params = []
    model_params += self.attentioner.parameters()

    self.optm = torch.optim.Adam(model_params, lr=learning_rate)

    # Restore Model
    if not self.args.restart:
      self.load_checkpoint()

    # Setup Global Train Index
    self.gidx = self.epoch * len(self.dataset_train)

    total_epochs = self.config['epochs']
    for _ in range(self.epoch, total_epochs):
      self.run_train()
      self.run_valid()

      self.epoch += 1

  def run_train(self):
    logging_interval = self.config['logging_interval']
    heavy_logging_interval = self.config['heavy_logging_interval']

    alpha = self.config['alpha']

    dslt_mt = utils.AverageMeter()  # DataSet Loading Time Meter
    btpt_mt = utils.AverageMeter()  # BaTch Processing Time Meter

    loss_mt = utils.AverageMeter()

    var_loss_mt = utils.AverageMeter()  # VARiance LOSS Meter
    ara_pnlt_mt = utils.AverageMeter()  # AReA PeNaLTy Meter

    var_rdpt_mt = utils.AverageMeter()  # VARiance ReDuction PercenTage Meter

    metrics = [dslt_mt, btpt_mt, loss_mt, var_loss_mt, ara_pnlt_mt, var_rdpt_mt]

    # switch to train mode
    self.attentioner.train()
    self.image_smoother.eval()
    self.variance_pool2d.eval()

    split_t = time.time()
    for step, (images, _) in enumerate(self.dataset_train):
      batch_size = images.size(0)

      # measure data loading time
      dslt_mt.update(time.time() - split_t)

      # data transfer
      images = images.cuda(non_blocking=True)

      # forward
      attens = self.attentioner(images)  # Generate Attention Map
      smthed = self.image_smoother(images)  # Calculate Smoothed Image

      weighed_images = attens * smthed + (1 - attens) * images

      # calculate loss
      weighed_images_var = self.variance_pool2d(weighed_images)
      images_var = self.variance_pool2d(images)
      smthed_var = self.variance_pool2d(smthed)

      variance_loss = torch.mean(weighed_images_var)
      # area_penalty ranges in [0, 1]. We decay the area penalty to get working attentions.
      # Punish more when closing to 1, less when closing to 0.
      area_penalty = torch.mean(attens)
      area_penalty = area_penalty**(3 - 2 * area_penalty)
      loss = alpha * variance_loss + (1 - alpha) * area_penalty

      # calculate metric helpers
      var_cur = torch.mean(weighed_images_var)
      var_min = torch.mean(smthed_var)
      var_max = torch.mean(images_var)
      variance_reduction = (var_max - var_cur) / (var_max - var_min)

      # update meters
      var_loss_mt.update(variance_loss.item(), batch_size)
      ara_pnlt_mt.update(area_penalty.item(), batch_size)
      var_rdpt_mt.update(variance_reduction.item(), batch_size)
      loss_mt.update(loss.item(), batch_size)

      # compute gradients and BP
      self.optm.zero_grad()
      loss.backward()
      self.optm.step()

      # meature time
      btpt_mt.update(time.time() - split_t)
      split_t = time.time()

      if step % logging_interval == 0:
        # Print console info
        self.logger.info(
            '[epoch=%04d/train]'
            '[%08d/%08d]'
            '[time=%6.3f]'
            '[dslt=%6.3f]'
            '[loss=%8.4f]'
            '[var_loss=%8.4f]'
            '[ara_pnlt=%8.4f]'
            '[var_rdpt=%05.1f%%]',
            self.epoch,
            step,
            len(self.dataset_train),
            btpt_mt.avg,
            dslt_mt.avg,
            loss_mt.avg,
            var_loss_mt.avg,
            ara_pnlt_mt.avg,
            var_rdpt_mt.avg * 100,
        )

        # Reset all metrics
        for metric in metrics:
          metric.reset()

      if step % heavy_logging_interval == 0:
        # Save demo images
        torchvision.utils.save_image(
            images.cpu(),  #
            str(self.images_path / f'train-images-{self.epoch:04d}-{step:08d}.jpg'))
        torchvision.utils.save_image(
            attens.cpu(),  #
            str(self.images_path / f'train-attens-{self.epoch:04d}-{step:08d}.png'))
        torchvision.utils.save_image(
            smthed.cpu(),  #
            str(self.images_path / f'train-smthed-{self.epoch:04d}-{step:08d}.jpg'))
        torchvision.utils.save_image(
            weighed_images.cpu(),  #
            str(self.images_path / f'train-weighed_images-{self.epoch:04d}-{step:08d}.jpg'))

      self.gidx += 1

  def run_valid(self):
    alpha = self.config['alpha']

    btpt_mt = utils.AverageMeter()  # BaTch Processing Time Meter

    loss_mt = utils.AverageMeter()

    var_loss_mt = utils.AverageMeter()  # VARiance LOSS Meter
    ara_pnlt_mt = utils.AverageMeter()  # AReA PeNaLTy Meter

    var_rdpt_mt = utils.AverageMeter()  # VARiance ReDuction PercenTage Meter

    # switch to evaluate mode
    self.attentioner.eval()
    self.image_smoother.eval()
    self.variance_pool2d.eval()

    with torch.no_grad():
      split_t = time.time()
      for step, (images, _) in enumerate(self.dataset_valid):
        batch_size = images.size(0)

        # data transfer
        images = images.cuda(non_blocking=True)

        # forward
        attens = self.attentioner(images)  # Generate Attention Map
        smthed = self.image_smoother(images)  # Calculate Smoothed Image

        weighed_images = attens * smthed + (1 - attens) * images

        # calculate loss
        weighed_images_var = self.variance_pool2d(weighed_images)
        images_var = self.variance_pool2d(images)
        smthed_var = self.variance_pool2d(smthed)

        variance_loss = torch.mean(weighed_images_var)
        # area_penalty ranges in [0, 1]. We decay the area penalty to get working attentions.
        # Punish more when closing to 1, less when closing to 0.
        area_penalty = torch.mean(attens)
        area_penalty = area_penalty**(3 - 2 * area_penalty)
        loss = alpha * variance_loss + (1 - alpha) * area_penalty

        # calculate metric helpers
        var_cur = torch.mean(weighed_images_var)
        var_min = torch.mean(smthed_var)
        var_max = torch.mean(images_var)
        variance_reduction = (var_max - var_cur) / (var_max - var_min)

        # update meters
        var_loss_mt.update(variance_loss.item(), batch_size)
        ara_pnlt_mt.update(area_penalty.item(), batch_size)
        var_rdpt_mt.update(variance_reduction.item(), batch_size)
        loss_mt.update(loss.item(), batch_size)

        # meature time
        btpt_mt.update(time.time() - split_t)
        split_t = time.time()

        if step < self.config['valid_images_save_num']:
          # Save demo images
          torchvision.utils.save_image(
              images.cpu(),  #
              str(self.images_path / f'valid-images-{self.epoch:04d}-{step:08d}.jpg'))
          torchvision.utils.save_image(
              attens.cpu(),  #
              str(self.images_path / f'valid-attens-{self.epoch:04d}-{step:08d}.png'))
          torchvision.utils.save_image(
              smthed.cpu(),  #
              str(self.images_path / f'valid-smthed-{self.epoch:04d}-{step:08d}.jpg'))
          torchvision.utils.save_image(
              weighed_images.cpu(),  #
              str(self.images_path / f'valid-weighed_images-{self.epoch:04d}-{step:08d}.jpg'))

    # Print console info
    self.logger.info(
        '[epoch=%04d/valid]'
        '[time=%6.3f]'
        '[loss=%8.4f]'
        '[var_loss=%8.4f]'
        '[ara_pnlt=%8.4f]'
        '[var_rdpt=%05.1f%%]',
        self.epoch,
        btpt_mt.avg,
        loss_mt.avg,
        var_loss_mt.avg,
        ara_pnlt_mt.avg,
        var_rdpt_mt.avg * 100,
    )

    # Save checkpoints
    if self.best_loss is None or loss_mt.avg < self.best_loss:
      self.save_checkpoint(
          best_metric_name='loss', metric_values={
              'epoch': self.epoch,
              'loss': loss_mt.avg,
              'var_loss': var_loss_mt.avg,
              'ara_pnlt': ara_pnlt_mt.avg,
              'var_rdpt': var_rdpt_mt.avg,
          })

    self.save_checkpoint()

    # Update metrics
    if self.best_loss is None:
      self.best_loss = loss_mt.avg
    else:
      self.best_loss = min(self.best_loss, loss_mt.avg)


if __name__ == '__main__':
  Main().run()
