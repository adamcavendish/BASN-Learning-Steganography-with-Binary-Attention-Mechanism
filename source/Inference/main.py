"""
Embed Process:
1. Variance Minimization Attention Map from Cover Image
2. Feature Similarity Attention Map from Cover Image
3. Create Fusion Attention Map
4. Embed Message into Attention Areas
5. Return Stego Image

Extract Process:
1. Variance Minimization Attention Map from Stego Image
2. Feature Similarity Attention Map from Stego Image
3. Create Fusion Attention Map
4. Extract Message from Attention Areas
5. Return Hidden Message
"""
# pylint: disable=E1101, E1102, C0414
import itertools
import multiprocessing as mp
import operator
import time
import typing as tp

import torch
import torch.nn
import torchvision.datasets
import torchvision.transforms as transforms

import cytoolz as tz
import numpy as np

from torchsummary import summary

import base_main
import dataset_tools
import utils

import vm_model
import fs_model


class DataObject(object):
  """Misc Data used in Training and Validation"""

  class Metrics(object):
    """All Metrics used in Training and Validation"""

    def __init__(self):
      self.dslt = utils.AverageMetrics()  # DataSet Loading Time
      self.btpt = utils.AverageMetrics()  # BaTch Processing Time

      self.cs1t = utils.AverageMetrics()  # Cover images Step 1 processing Time
      self.cs2t = utils.AverageMetrics()  # Cover images Step 2 processing Time
      self.ebpt = utils.AverageMetrics()  # EmBedding Processing Time

      self.ss1t = utils.AverageMetrics()  # Stego images Step 1 processing Time
      self.ss2t = utils.AverageMetrics()  # Stego images Step 2 processing Time
      self.expt = utils.AverageMetrics()  # EXtracting Processing Time

      self.s1al = utils.AverageMetrics()  # Step 1 Attention Loss
      self.s2al = utils.AverageMetrics()  # Step 2 Attention Loss
      self.s3al = utils.AverageMetrics()  # Step 3 Attention Loss

      self.c1aa = utils.AverageMetrics()  # Cover step 1 Attention Area
      self.c2aa = utils.AverageMetrics()  # Cover step 2 Attention Area
      self.c3aa = utils.AverageMetrics()  # Cover step 3 Attention Area

      self.s1aa = utils.AverageMetrics()  # Stego step 1 Attention Area
      self.s2aa = utils.AverageMetrics()  # Stego step 2 Attention Area
      self.s3aa = utils.AverageMetrics()  # Stego step 3 Attention Area

      self.msg_diff = utils.AverageMetrics()  # MeSsaGe bitwise DIFFerence
      self.msg_edis = utils.AverageMetrics()  # MeSsaGe bitwise Editing DIStance

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

      self.msg_diff = None  # MeSsaGe bitwise DIFFerence
      self.msg_edis = None  # MeSsaGe bitwise Editing DIStance

  class Debug(object):
    """Debug Objects"""

    def __init__(self):
      """
      Here embd_* stands for debugging information retrieved during embedding process
      where extt_* stands for extraction process.

      pre_ILSBN stands for dirty image/message preceding the ILSBN process
      suc_ILSBN stands for clean image/message succeeding the ILSBN process
      """
      self.embd_cover_images = None
      self.embd_cover_attens = None

      self.embd_lsb_msgs_pre_ILSBN = None
      self.embd_lsb_msgs_suc_ILSBN = None

      self.embd_cover_images_LSBN = None

      self.embd_stego_images_pre_ILSBN = None
      self.embd_stego_images_suc_ILSBN = None

      self.extt_stego_images = None
      self.extt_stego_attens = None

      self.extt_lsb_msgs_pre_ILSBN = None
      self.extt_lsb_msgs_suc_ILSBN = None

  def __init__(self):
    self.mt = DataObject.Metrics()
    self.ls = DataObject.Loss()
    self.dbg = DataObject.Debug()

    # Cover Images: torch.Tensor
    self.cover_images = None
    # Cover's Variance Minimization Attention Map: torch.Tensor (same shape as cover_images)
    self.cover_vm_attens = None
    # Cover's Feature Similarity Attention Map: torch.Tensor (same shape as cover_images)
    self.cover_fs_attens = None
    # Cover's Fusion Attention Map: torch.Tensor (same shape as cover_images)
    self.cover_fusion_attention = None

    # Stego Images (Use Fusion Attention for Embedding): torch.Tensor (same shape as cover_images)
    self.stego_images = None
    # Stego's Variance Minimization Attention Map: torch.Tensor (same shape as cover_images)
    self.stego_vm_attens = None
    # Stego's Feature Similarity Attention Map: torch.Tensor (same shape as cover_images)
    self.stego_fs_attens = None
    # Stego's Fusion Attention Map: torch.Tensor (same shape as cover_images)
    self.stego_fusion_attention = None

    self.filenames = None  # List[str], len() == batch_size

    self.embedded_messages = None  # List[np.ndarray[np.uint8]], len() == batch_size
    self.extracted_messages = None  # List[np.ndarray[np.uint8]], len() == batch_size


class Main(base_main.BaseMain):
  """Main Module"""

  def __init__(self, default_config='config-inference.yaml'):
    super(Main, self).__init__(default_config)

    self.dataset_valid = self.prepare_datasets()
    self.hidden_message = self.prepare_message()

    # models
    self.vm_attentioner: torch.nn.Module = None
    self.fs_attentioner: torch.nn.Module = None

    FusionFunc = tp.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    self.fusion_algorithms: tp.Dict[FusionFunc] = self.prepare_fusion_algorithms()

    # logging info
    self.valid = DataObject()

  def prepare_fusion_algorithms(self):
    """Prepare fusion algorithms"""

    def _min(vm_attens, fs_attens):
      return torch.min(input=vm_attens, other=fs_attens)

    def _max(vm_attens, fs_attens):
      return torch.max(input=vm_attens, other=fs_attens)

    def _mean(vm_attens, fs_attens):
      return (vm_attens + fs_attens) / 2

    return {
        'min': _min,
        'max': _max,
        'mean': _mean,
    }

  def prepare_datasets(self):
    """Prepare datasets"""
    images_path = self.config['input_images_path']

    dataset_valid = dataset_tools.ImageFolderFN(
        images_path, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    dataset_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=self.config['batch_size'],
        shuffle=False,
        num_workers=self.config['dataset_worker_num'],
        pin_memory=True)

    return dataset_valid

  def prepare_message(self):
    """Prepare hidden message"""
    message_path = self.config['hidden_message_path']

    with open(message_path, 'rb') as f:
      message = f.read()

    return np.array(list(message), dtype=np.uint8)

  def load_pretrained_models(self):
    """Load up pretrained models"""
    # Variance Minimization
    abspath = str(self.pretrain_path / self.config['checkpoint_step_1'])
    self.logger.info('Load Step-1 Variance Minimization from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.vm_attentioner.load_state_dict(checkpoint)

    # Feature Similarity
    abspath = str(self.pretrain_path / self.config['checkpoint_step_2'])
    self.logger.info('Load Step-2 Feature Similarity from checkpoint: %s', abspath)
    checkpoint = torch.load(abspath)
    checkpoint = utils.remove_data_parallel_prefix(checkpoint)
    self.fs_attentioner.load_state_dict(checkpoint)

    self.logger.info('Pretrained checkpoints loaded.')

  def run(self):
    """Run, main entry"""
    batch_size = self.config['batch_size']

    self.vm_attentioner = vm_model.Attentioner().cuda()
    self.fs_attentioner = fs_model.attentioner().cuda()

    # Change to eval mode
    self.vm_attentioner.eval()
    self.fs_attentioner.eval()

    self.logger.debug('Variance Minimization Attentioner Architecture')
    summary(self.vm_attentioner, (3, 224, 224), batch_size=batch_size)

    self.logger.debug('Feature Similarity Attentioner Architecture')
    summary(self.fs_attentioner, (3, 224, 224), batch_size=batch_size)

    # Load Pretrained Models
    self.load_pretrained_models()

    split_t = time.time()
    with torch.no_grad():
      for step, (cover_images, _, filenames) in enumerate(self.dataset_valid):
        # measure data loading time
        self.valid.mt.dslt.update(time.time() - split_t)

        # data transfer
        self.valid.cover_images = cover_images.cuda()
        self.valid.filenames = filenames

        # run one batch
        self.run_once()

        # meature batch processing time
        self.valid.mt.btpt.update(time.time() - split_t)
        split_t = time.time()

        # Save Results
        self.image_saving(step=step)
        self.image_saving_detailed(step=step)

        # Logging
        self.logging_once(step=step)

  def run_once(self):
    """Run One Step"""
    do = self.valid

    batch_size = do.cover_images.shape[0]

    # Generate cover attentions
    with utils.metrics_update(do.mt.cs1t, batch_size):
      do.cover_vm_attens = self.vm_attentioner(do.cover_images)
    with utils.metrics_update(do.mt.cs2t, batch_size):
      do.cover_fs_attens = self.fs_attentioner(do.cover_images)

    # Fuse together cover attentions
    f_fusion = self.fusion_algorithms[self.config['fusion_algorithm']]
    do.cover_fusion_attention = f_fusion(do.cover_vm_attens, do.cover_fs_attens)

    # Cover Images: Clamp, Change shape:
    #   tensor(batch, 3, 224, 224) -> np.ndarray(batch, 3*224*224, np.uint8)
    cover_images = torch.clamp(do.cover_images * 255, min=0, max=255).view(batch_size, -1)
    cover_images = cover_images.cpu().detach().numpy().astype(np.uint8)
    # Cover Attentions: Round, Change shape:
    #   tensor(batch, 3, 224, 224) -> np.ndarray(batch, 3*224*224, np.uint8)
    cover_attentions = Main.create_attention_bitnum(do.cover_fusion_attention)
    cover_attentions = cover_attentions.cpu().detach().numpy().astype(np.uint8)

    # Generate same hidden messages for each cover image
    hidden_messages = [self.hidden_message] * batch_size

    def _get_embedded_bytes_num(attention: np.ndarray, ILSBN: int):
      """Calculate the num of bytes for embedding according to attention capacity

      attention: the attention capacity array used on images
      ILSBN: Ignore Num of LSB Bits

      ILSBN is removed from attention capacity
      """
      attention_ILSBN = Main.create_attention_ILSBN(attention, ILSBN)
      ret = int(np.ceil(np.sum(attention_ILSBN) / 8))
      # drop last byte since they are often incomplete
      ret = ret - 1 if ret > 1 else ret
      return ret

    # Calculate the actual embedded message according to attention capacity
    ILSBN = self.config['ignore_lsb_num']
    do.embedded_messages = [
        msg[:_get_embedded_bytes_num(atten, ILSBN)]
        for msg, atten in zip(hidden_messages, cover_attentions)
    ]

    # Embed messages into cover images
    with utils.metrics_update(do.mt.ebpt, batch_size):
      stego_images = self.adaptive_batch_lsbr_embed(  #
          cover_images, cover_attentions, hidden_messages)

    # Preserve stego_images as torch tensor
    do.stego_images = torch.tensor(stego_images, dtype=torch.float32).cuda() / 255.
    do.stego_images = do.stego_images.view_as(do.cover_images)

    # Generate stego attentions
    with utils.metrics_update(do.mt.ss1t, batch_size):
      do.stego_vm_attens = self.vm_attentioner(do.stego_images)
    with utils.metrics_update(do.mt.ss2t, batch_size):
      do.stego_fs_attens = self.fs_attentioner(do.stego_images)

    # Fuse together stego attentions
    f_fusion = self.fusion_algorithms[self.config['fusion_algorithm']]
    do.stego_fusion_attention = f_fusion(do.stego_vm_attens, do.stego_fs_attens)

    # Stego Attentions: Round, Change shape:
    #   tensor(batch, 3, 224, 224) -> np.ndarray(batch, 3*224*224, np.uint8)
    stego_attentions = Main.create_attention_bitnum(do.stego_fusion_attention)
    stego_attentions = stego_attentions.cpu().detach().numpy().astype(np.uint8)

    # Extract messages out
    with utils.metrics_update(do.mt.expt, batch_size):
      do.extracted_messages = self.adaptive_batch_lsbr_extract(  #
          stego_images, stego_attentions)

    # Calculate losses
    do.ls.s1al = torch.abs(
        Main.create_attention_bitnum(do.cover_vm_attens) -
        Main.create_attention_bitnum(do.stego_vm_attens)).sum(dim=1).mean()
    do.ls.s2al = torch.abs(
        Main.create_attention_bitnum(do.cover_fs_attens) -
        Main.create_attention_bitnum(do.stego_fs_attens)).sum(dim=1).mean()
    do.ls.s3al = torch.abs(
        Main.create_attention_bitnum(do.cover_fusion_attention) -
        Main.create_attention_bitnum(do.stego_fusion_attention)).sum(dim=1).mean()

    do.ls.c1aa = do.cover_vm_attens.mean()
    do.ls.c2aa = do.cover_fs_attens.mean()
    do.ls.c3aa = do.cover_fusion_attention.mean()

    do.ls.s1aa = do.stego_vm_attens.mean()
    do.ls.s2aa = do.stego_fs_attens.mean()
    do.ls.s3aa = do.stego_fusion_attention.mean()

    with mp.Pool() as pool:
      msg_diff = pool.starmap(Main.bitwise_difference,
                              zip(do.embedded_messages, do.extracted_messages))
    do.ls.msg_diff = np.mean(msg_diff)

    do.ls.msg_edis = Main.bitwise_edit_difference(cover_attentions, stego_attentions, ILSBN)

    # Update loss metrics
    do.mt.s1al.update(do.ls.s1al, batch_size)
    do.mt.s2al.update(do.ls.s2al, batch_size)
    do.mt.s3al.update(do.ls.s3al, batch_size)

    do.mt.c1aa.update(do.ls.c1aa, batch_size)
    do.mt.c2aa.update(do.ls.c2aa, batch_size)
    do.mt.c3aa.update(do.ls.c3aa, batch_size)

    do.mt.s1aa.update(do.ls.s1aa, batch_size)
    do.mt.s2aa.update(do.ls.s2aa, batch_size)
    do.mt.s3aa.update(do.ls.s3aa, batch_size)

    do.mt.msg_diff.update(do.ls.msg_diff, batch_size)
    do.mt.msg_edis.update(do.ls.msg_edis, batch_size)

    if self.config['debug']:
      # pylint: disable=C0301,C0326
      # yapf: disable
      Main.attention_debug(
          self.logger, [
              ('embd/stego pILSBN  ', do.dbg.embd_stego_images_pre_ILSBN[0], do.dbg.embd_cover_attens[0]),
              ('embd/stego sILSBN  ', do.dbg.embd_stego_images_suc_ILSBN[0], do.dbg.embd_cover_attens[0]),
              ('embd/cover LSBN    ', do.dbg.embd_cover_images_LSBN[0],      None),
              ('embd/cover         ', do.dbg.embd_cover_images[0],           None),
              ('extt/stego pILSBN  ', do.dbg.extt_stego_images[0],           do.dbg.extt_stego_attens[0]),
              ('embd/LSB-msg pILSBN', do.dbg.embd_lsb_msgs_pre_ILSBN[0],     do.dbg.embd_cover_attens[0]),
              ('extt/LSB-msg pILSBN', do.dbg.extt_lsb_msgs_pre_ILSBN[0],     do.dbg.extt_stego_attens[0]),
              ('embd/LSB-msg sILSBN', do.dbg.embd_lsb_msgs_suc_ILSBN[0],     do.dbg.embd_cover_attens[0]),
              ('extt/LSB-msg sILSBN', do.dbg.extt_lsb_msgs_suc_ILSBN[0],     do.dbg.extt_stego_attens[0]),
          ],
          max_elem_line=20,
          num_elems=20)
      # yapf: enable

  @staticmethod
  def create_attention_ILSBN(attention: np.ndarray, ILSBN: int):
    """Generate attention with ILSBN removed

    Here 'attention' is the embedding capacity (num of bits embedded)
    where last 'ILSBN' bits should be recovered to cover images' LSBs,
    and therefore, only 'attention - ILSBN' bits can be embedded
    """
    attention_ILSBN = attention.astype(np.int8) - ILSBN
    attention_ILSBN[attention_ILSBN < 0] = 0
    attention_ILSBN = attention_ILSBN.astype(np.uint8)
    return attention_ILSBN

  @staticmethod
  def create_attention_bitnum(attention):
    """Generate from float attention to attention bits (range from 0 - 8)"""
    return torch.round(attention * 8).view(attention.size(0), -1)

  @staticmethod
  def partition(plist, sizes):
    """Partition a list according to sizes"""
    plist_iter = iter(plist)
    return [list(tz.take(s, plist_iter)) for s in sizes]

  @staticmethod
  def create_attention_lsb_message(message, attention, ILSBN):
    """Create lsb hidden messages based on ILSBN and attention of each cover"""
    assert isinstance(message, np.ndarray)
    assert message.dtype == np.uint8

    assert isinstance(attention, np.ndarray)
    assert attention.dtype == np.uint8
    assert np.min(attention) >= 0 and np.max(attention) <= 8

    # Here 'attention' is the embedding capacity (num of bits embedded),
    # however we only need 'attention - ILSBN' bits of messages
    # since last 'ILSBN' bits should be recovered to cover images' LSBs
    # (* Note: cast from uint8 to int8 to prevent 0 - 1 -> 255 *)
    attention_ILSBN = attention.astype(np.int8) - ILSBN
    attention_ILSBN[attention_ILSBN < 0] = 0
    attention_ILSBN = attention_ILSBN.astype(np.uint8)

    # Unpack and partition into lists of attention bits
    message = np.unpackbits(message)
    message = Main.partition(message, attention_ILSBN)
    # Filling zeros at front to pad message bits aligned as bytes
    message = [(8 - len(part)) * [0] + part for part in message]
    message = np.packbits(message)
    # Left shift to reserve space for cover images LSBs recovery
    message = np.left_shift(message, ILSBN)

    return message

  def adaptive_batch_lsbr_embed(self, cover_images, attentions, messages):
    """Embedding process"""
    assert isinstance(cover_images, np.ndarray)
    assert cover_images.dtype == np.uint8

    assert isinstance(attentions, np.ndarray)
    assert attentions.dtype == np.uint8
    assert np.min(attentions) >= 0 and np.max(attentions) <= 8

    do = self.valid
    batch_size = cover_images.shape[0]

    assert len(messages) == batch_size

    ILSBN = self.config['ignore_lsb_num']

    # Generate LSB messages for the batch
    lsb_messages = []
    for bidx in range(batch_size):
      attention = attentions[bidx]
      message = messages[bidx]
      lsb_message = Main.create_attention_lsb_message(message, attention, ILSBN)
      lsb_messages.append(lsb_message)
    lsb_messages = np.array(lsb_messages, dtype=np.uint8)

    # Embed messages at attentions
    stego_images_pre_ILSBN = np.right_shift(cover_images, attentions)
    stego_images_pre_ILSBN = np.left_shift(stego_images_pre_ILSBN, attentions)
    stego_images_pre_ILSBN = np.bitwise_or(stego_images_pre_ILSBN, lsb_messages)

    # Takeout cover images' LSBs (ILSBN bits/pixel)
    ILSBN_mask = np.uint8(np.uint8(0xFF) >> ILSBN << ILSBN)
    cover_images_LSBN = np.bitwise_and(cover_images, np.bitwise_not(np.uint8(ILSBN_mask)))

    # Restore the stego images' LSBs to cover images (ILSBN bits/pixel)
    stego_images_suc_ILSBN = stego_images_pre_ILSBN
    stego_images_suc_ILSBN = np.bitwise_and(stego_images_suc_ILSBN, ILSBN_mask)
    stego_images_suc_ILSBN = np.bitwise_or(stego_images_suc_ILSBN, cover_images_LSBN)

    if self.config['debug']:
      # Prepare information for attention-debugging
      do.dbg.embd_cover_images = cover_images
      do.dbg.embd_cover_attens = attentions
      do.dbg.embd_lsb_msgs_pre_ILSBN = lsb_messages
      do.dbg.embd_lsb_msgs_suc_ILSBN = np.right_shift(lsb_messages, ILSBN)
      do.dbg.embd_cover_images_LSBN = cover_images_LSBN
      do.dbg.embd_stego_images_pre_ILSBN = stego_images_pre_ILSBN
      do.dbg.embd_stego_images_suc_ILSBN = stego_images_suc_ILSBN

    stego_images = stego_images_suc_ILSBN
    return stego_images

  def adaptive_batch_lsbr_extract(self, stego_images, attentions):
    """Extraction process"""
    assert isinstance(stego_images, np.ndarray)
    assert stego_images.dtype == np.uint8

    assert isinstance(attentions, np.ndarray)
    assert attentions.dtype == np.uint8
    assert np.min(attentions) >= 0 and np.max(attentions) <= 8

    do = self.valid
    batch_size = stego_images.shape[0]

    ILSBN = self.config['ignore_lsb_num']
    BITS_END = 8 - ILSBN

    if self.config['debug']:
      # Here 'attention' is the embedding capacity (num of bits embedded)
      # so the mask is '2**attentions - 1'
      # i.e. attention = 3 is that 3 bits were embedded
      #      lsb_mask would be 0b0000'0111 being 7, aka. '2**3 - 1'
      lsb_masks = 2**attentions - 1

      # Extract messages together with cover images' LSBs (ILSBN bits/pixel)
      lsb_messages_pre_ILSBN = np.bitwise_and(stego_images, lsb_masks)

      # Extract the ILSBN messages part, aka. without cover images' LSBs
      lsb_messages_suc_ILSBN = np.right_shift(lsb_messages_pre_ILSBN, ILSBN)

      # Prepare information for attention-debugging
      do.dbg.extt_stego_images = stego_images
      do.dbg.extt_stego_attens = attentions
      do.dbg.extt_lsb_msgs_pre_ILSBN = lsb_messages_pre_ILSBN
      do.dbg.extt_lsb_msgs_suc_ILSBN = lsb_messages_suc_ILSBN

    # Unpack into bit stream and group into groups of 8 bits (a byte)
    messages = np.unpackbits(stego_images, axis=1)
    messages = messages.reshape((batch_size, -1, 8))

    hidden_messages = []
    for bidx in range(batch_size):
      message = messages[bidx]
      attention = attentions[bidx]

      # Here 'attention' is the embedding capacity
      # then, information lies in '8 - attention' least bits
      lsb_attention = 8 - attention

      # Takeout real message bits (with ILSBN bits/pixel removed)
      lsb_stream = [
          list(byte_bits[lsb_atten:BITS_END])
          for byte_bits, lsb_atten in zip(message, lsb_attention)
      ]
      # Chain concat / flatten lists to make bit stream continuous
      lsb_stream = list(itertools.chain.from_iterable(lsb_stream))
      # Pack bit stream into bytes
      lsb_stream = np.packbits(lsb_stream)

      # Drop last byte since they are often incomplete
      hidden_messages.append(lsb_stream[:-1])

    return hidden_messages

  @staticmethod
  def bitwise_difference(lhs, rhs):
    """Calculate bitwise difference between two byte arrays"""
    assert isinstance(lhs, np.ndarray) and isinstance(rhs, np.ndarray)
    assert lhs.dtype == np.uint8 and rhs.dtype == np.uint8

    lhs, rhs = lhs.flatten(), rhs.flatten()

    # Turn into same shape
    truncated_bits_num = 0
    if len(lhs) > len(rhs):
      lhs = lhs[:len(rhs)]
      truncated_bits_num = len(lhs) - len(rhs)
    elif len(lhs) < len(rhs):
      rhs = rhs[:len(lhs)]
      truncated_bits_num = len(rhs) - len(lhs)

    lhs = np.unpackbits(lhs).astype(np.int8)
    rhs = np.unpackbits(rhs).astype(np.int8)

    return np.sum(np.abs(lhs - rhs)) + truncated_bits_num

  @staticmethod
  def bitwise_edit_difference(cover_attentions: np.ndarray, stego_attentions: np.ndarray,
                              ILSBN: int):
    """Calculate bitwise edit difference using attentions

    Note: remember to process the conversion.
          np.uint8 would yield '0 - 1 = 255'
    """
    cover_attentions = Main.create_attention_ILSBN(cover_attentions, ILSBN).astype(np.int8)
    stego_attentions = Main.create_attention_ILSBN(stego_attentions, ILSBN).astype(np.int8)
    return np.abs(cover_attentions - stego_attentions).sum(axis=1).mean()

  def image_saving(self, step):
    """Save a batch into one image"""
    do = self.valid

    torchvision.utils.save_image(
        do.cover_images.cpu(),  #
        str(self.images_path / f'cover-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_vm_attens.cpu(),  #
        str(self.images_path / f'cover-vm-attens-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_fs_attens.cpu(),  #
        str(self.images_path / f'cover-fs-attens-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.cover_fusion_attention.cpu(),  #
        str(self.images_path / f'cover-fusion-attention-{step:08d}.jpg'))

    torchvision.utils.save_image(
        do.stego_images.cpu(),  #
        str(self.images_path / f'stego-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.stego_vm_attens.cpu(),  #
        str(self.images_path / f'stego-vm-attens-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.stego_fs_attens.cpu(),  #
        str(self.images_path / f'stego-fs-attens-{step:08d}.jpg'))
    torchvision.utils.save_image(
        do.stego_fusion_attention.cpu(),  #
        str(self.images_path / f'stego-fusion-attention-{step:08d}.jpg'))

  def image_saving_detailed(self, step):
    """Save a batch as a directory of images"""
    do = self.valid

    name = f'cover-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_images.cpu(), path, name)

    name = f'cover-vm-attens-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_vm_attens.cpu(), path, name)

    name = f'cover-fs-attens-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_fs_attens.cpu(), path, name)

    name = f'cover-fusion-attention-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.cover_fusion_attention.cpu(), path, name)

    name = f'stego-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_images.cpu(), path, name)

    name = f'stego-vm-attens-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_vm_attens.cpu(), path, name)

    name = f'stego-fs-attens-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_fs_attens.cpu(), path, name)

    name = f'stego-fusion-attention-{step:08d}'
    path = self.images_path / name
    path.mkdir(parents=True, exist_ok=True)
    utils.save_image_batch(do.stego_fusion_attention.cpu(), path, name)

  def logging_once(self, step):
    """Log information at one step"""
    do = self.valid

    # Print console info
    self.logger.info(
        '[step=%04d]'
        '[dslt=%6.3f]'
        '[btpt=%6.3f],'
        '[cs1t=%6.3f]'
        '[cs2t=%6.3f]'
        '[ebpt=%6.3f],'
        '[ss1t=%6.3f]'
        '[ss2t=%6.3f]'
        '[expt=%6.3f]',  #
        step,
        do.mt.dslt.avg,
        do.mt.btpt.avg,
        do.mt.cs1t.avg,
        do.mt.cs2t.avg,
        do.mt.ebpt.avg,
        do.mt.ss1t.avg,
        do.mt.ss2t.avg,
        do.mt.expt.avg)
    self.logger.info(
        '[step=%04d]'
        '[step-1-attn-loss=%8.1fb]'
        '[step-2-attn-loss=%8.1fb]'
        '[step-3-attn-loss=%8.1fb]'
        '[step-1-area=%5.3f/%5.3f]'
        '[step-2-area=%5.3f/%5.3f]'
        '[step-3-area=%5.3f/%5.3f]',  #
        step,
        do.mt.s1al.avg,
        do.mt.s2al.avg,
        do.mt.s3al.avg,
        do.mt.c1aa.avg,
        do.mt.s1aa.avg,
        do.mt.c2aa.avg,
        do.mt.s2aa.avg,
        do.mt.c3aa.avg,
        do.mt.s3aa.avg)

    avg_embd_bytes = np.mean([msg.size for msg in do.embedded_messages])
    avg_extt_bytes = np.mean([msg.size for msg in do.extracted_messages])
    total_pixels = tz.reduce(operator.mul, do.cover_images.shape[2:])
    self.logger.info(
        '[embd-payload=%6db/%6dP(%5.2f)]'
        '[extt-payload=%6db/%8dP(%5.2f)],'
        '[msg-diff=%10.3fb/%10db(%05.2f%%)]'
        '[msg-EDis=%10.3fb/%10db(%05.2f%%)]',  #
        avg_embd_bytes * 8,
        total_pixels,
        avg_embd_bytes * 8 / total_pixels,
        avg_extt_bytes * 8,
        total_pixels,
        avg_extt_bytes * 8 / total_pixels,
        do.mt.msg_diff.avg,
        avg_embd_bytes * 8,
        do.mt.msg_diff.avg / (avg_embd_bytes * 8) * 100,
        do.mt.msg_edis.avg,
        avg_embd_bytes * 8,
        do.mt.msg_edis.avg / (avg_embd_bytes * 8) * 100)

  @staticmethod
  def attention_debug(logger, image_atten_list, max_elem_line=20, num_elems=100, delim=' '):
    """Debug Cover Data and Attention Data

    logger: python logger
    image_atten_list: List[Tuple[str, np.ndarray, np.ndarray?]]:
      A list of (name, image, attention) tuples. attention can be None.
    max_elem_line: max number of elements in a line
    num_elems: max number of elements shown in console
    delim: delimeter used in between elements
    """
    max_name_len = 0
    for name, image, atten in image_atten_list:
      assert isinstance(name, str)

      assert isinstance(image, np.ndarray)
      assert len(image.shape) == 1
      assert image.dtype == np.uint8

      if not atten is None:
        assert isinstance(atten, np.ndarray)
        assert len(atten.shape) == 1
        assert atten.dtype == np.uint8

      num_elems = min(num_elems, image.size)
      max_name_len = max(max_name_len, len(name))

    def _str_elem(pixel, atten=None):
      atten = str(atten) if not atten is None else '-'
      return '%03d/%s' % (pixel, atten)

    num_indices = int(np.ceil(num_elems / max_name_len)) - 1
    for idx in range(num_indices):
      beg, end = idx * max_elem_line, (idx + 1) * max_elem_line

      for name, image, atten in image_atten_list:
        if not atten is None:
          sub_image = image[beg:end]
          sub_atten = atten[beg:end]

          dbg_info = delim.join([_str_elem(p, a) for p, a in zip(sub_image, sub_atten)])
          logger.debug('%s %s', name.ljust(max_name_len), dbg_info)
        else:
          sub_image = image[beg:end]

          dbg_info = delim.join([_str_elem(p) for p in sub_image])
          logger.debug('%s %s', name.ljust(max_name_len), dbg_info)
      logger.debug('')


if __name__ == '__main__':
  Main().run()
