'''
BaseMain Module
'''

import argparse
import random
import datetime
import pathlib
import logging

import numpy as np

import torch
import torch.backends.cudnn as cudnn

from ruamel.yaml import YAML


class BaseMain(object):
  '''Base Main Module'''

  def __init__(self, default_config):
    self.args, self.config = BaseMain.prepare_cmd_args(default_config)
    self.uuid = self.generate_uuid()
    self.prepare_seed_and_cudnn()

    self.logging_path = None
    self.images_path = None
    self.checkpoint_path = None
    self.prepare_directories()

    self.logger = self.prepare_loggers()

  @staticmethod
  def prepare_cmd_args(default_config):
    '''Prepare cmdline Arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_config, help='configuration file path')
    parser.add_argument('--comment', default='', help='current training comment')
    parser.add_argument('--restart', action='store_true', default=False)
    args = parser.parse_args()

    config_path = args.config

    # Setup configurations
    with open(config_path, 'r') as f:
      yaml = YAML()
      config = yaml.load(f)

    return args, config

  def generate_uuid(self):
    '''Generate Run UUID'''

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
    '''Make directories'''
    self.logging_path = pathlib.Path(self.config['logging_path'].format(uuid=self.uuid))
    self.logging_path.mkdir(parents=True, exist_ok=True)

    self.images_path = pathlib.Path(self.config['images_path'].format(uuid=self.uuid))
    self.images_path.mkdir(parents=True, exist_ok=True)

    self.checkpoint_path = pathlib.Path(self.config['checkpoint_path'].format(uuid=self.uuid))
    self.checkpoint_path.mkdir(parents=True, exist_ok=True)

    self.pretrain_path = pathlib.Path(self.config['pretrain_path'].format(uuid=self.uuid))
    self.pretrain_path.mkdir(parents=True, exist_ok=True)

    self.tensorboard_path = pathlib.Path(self.config['tensorboard_path'].format(uuid=self.uuid))
    self.tensorboard_path.mkdir(parents=True, exist_ok=True)

  def prepare_loggers(self):
    '''Create Loggers'''
    logging.basicConfig(
        format='[%(asctime)s][%(levelname)-5.5s] %(message)s',
        handlers=[
            logging.FileHandler(str(self.logging_path / f'{self.uuid}.log')),
            logging.StreamHandler(),
        ])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    return logger

  def prepare_seed_and_cudnn(self):
    '''Fix Seed'''
    seed = self.config['seed']
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if self.config['deterministic']:
      cudnn.deterministic = True
