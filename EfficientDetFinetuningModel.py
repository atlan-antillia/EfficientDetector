# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2021/02/10
# Copyright 2021 atlan@antillia.com All Rights Reserved.

# This EfficientDetFinetuningModel class is based on automl/efficientdet/main.py

# EfficientDetFinetuningModel.py


import multiprocessing
import os
from absl import app
from absl import flags

from absl import logging
import numpy as np

from tensorflow.python.ops import custom_gradient # pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops # pylint:disable=g-direct-tensorflow-import


def get_variable_by_name(var_name):
  """Given a variable name, retrieves a handle on the tensorflow Variable."""

  global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)

  def _filter_fn(item):
    try:
      return var_name == item.op.name
    except AttributeError:
      # Collection items without operation are ignored.
      return False

  candidate_vars = list(filter(_filter_fn, global_vars))

  if len(candidate_vars) >= 1:
    # Filter out non-trainable variables.
    candidate_vars = [v for v in candidate_vars if v.trainable]
  else:
    raise ValueError("Unsuccessful at finding variable {}.".format(var_name))

  if len(candidate_vars) == 1:
    return candidate_vars[0]
  elif len(candidate_vars) > 1:
    raise ValueError(
      "Unsuccessful at finding trainable variable {}. "
      "Number of candidates: {}. "
      "Candidates: {}".format(var_name, len(candidate_vars), candidate_vars))
  else:
    # The variable is not trainable.
    return None

custom_gradient.get_variable_by_name = get_variable_by_name

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os
import sys
import traceback

import dataloader
import det_model_fn
import hparams_config
import utils

from TrainConfigParser       import TrainConfigParser
from mAPEarlyStopping        import mAPEarlyStopping
from EvaluationResultsWriter import EvaluationResultsWriter
from EpochChangeNotifier     import EpochChangeNotifier
from TrainingLossesWriter    import TrainingLossesWriter

class EfficientDetFinetuningModel:

  def __init__(self, train_config):
    self.TRAIN          = 'train'
    self.EVAL           = 'eval'
    self.TRAIN_AND_EVAL = 'train_and_eval'
    
    self.parser         = TrainConfigParser(train_config)
    self.model_dir      = self.parser.model_dir()
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
   
    training_losses_file           = self.parser.training_losses_file()

    self.training_losses_writer    = TrainingLossesWriter(training_losses_file)

    evaluation_results_file        = self.parser.evaluation_results_file()
   
    self.evaluation_results_writer = EvaluationResultsWriter(evaluation_results_file)
    
    patience            = self.parser.early_stopping_patience()
    self.earyl_stopping = None
    if patience > 0:
      self.early_stopping = mAPEarlyStopping(patience=patience, verbose=1)
      
    if self.parser.strategy() == 'tpu':
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          self.parser.tpu(), zone=self.parser.tpu_zone(), project=self.parser.gcp_project() )
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      tpu_cluster_resolver = None

    # Check data path
    if self.parser.training_mode() in (self.TRAIN, self.TRAIN_AND_EVAL):
      if self.parser.training_file_pattern() is None:
        raise RuntimeError('Must specify --file_pattern for train.')
    
    if self.parser.training_mode() in (self.EVAL,  self.TRAIN_AND_EVAL):
      if self.parser.validation_file_pattern() is None:
        raise RuntimeError('Must specify --file_pattern for valid.')

    # Parse and override hparams
    self.config = hparams_config.get_detection_config(self.parser.model_name() )
    self.config.override(self.parser.hparams() )
    
    if self.parser.epochs():  # NOTE: remove this flag after updating all docs.
      self.config.num_epochs = self.parser.epochs()

    # Parse image size in case it is in string format.
    self.config.image_size = utils.parse_image_size(self.config.image_size)

    # The following is for spatial partitioning. `features` has one tensor while
    # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
    # partition is performed on `features` and all partitionable tensors of
    # `labels`, see the partition logic below.
    # In the TPUEstimator context, the meaning of `shard` and `replica` is the
    # same; follwing the API, here has mixed use of both.
    if self.parser.use_spatial_partition():
      print("=== use_spatical_partion")
      # Checks input_partition_dims agrees with num_cores_per_replica.
      if self.parser.cores_per_replica() != np.prod(self.parser.input_partition_dims() ):
        raise RuntimeError('--num_cores_per_replica must be a product of array'
                           'elements in --input_partition_dims.')

      labels_partition_dims = {
          'mean_num_positives': None,
          'source_ids':         None,
          'groundtruth_data':   None,
          'image_scales':       None,
          'image_masks':        None,
      }
      
      # The Input Partition Logic: We partition only the partition-able tensors.
      feat_sizes = utils.get_feat_sizes(
          self.config.get('image_size'), self.config.get('max_level'))
      for level in range(self.config.get('min_level'), self.config.get('max_level') + 1):

        def _can_partition(spatial_dim):
          partitionable_index = np.where(
              spatial_dim % np.array(self.parser.input_partition_dims() ) == 0)
          return len(partitionable_index[0]) == len(self.parser.input_partition_dims() )

        spatial_dim = feat_sizes[level]
        if _can_partition(spatial_dim['height']) and _can_partition(
            spatial_dim['width']):
          labels_partition_dims['box_targets_%d' %
                                level] = self.parser.input_partition_dims()
          labels_partition_dims['cls_targets_%d' %
                                level] = self.parser.input_partition_dims()
        else:
          labels_partition_dims['box_targets_%d' % level] = None
          labels_partition_dims['cls_targets_%d' % level] = None
      num_cores_per_replica = self.parser.cores_per_replica()
      input_partition_dims  = [self.parser.input_partition_dims(), labels_partition_dims]
      num_shards            = self.parser.training_cores() // num_cores_per_replica
    else:
      print("=== Not use_spatical_partion")
      num_cores_per_replica = None
      input_partition_dims  = None
      num_shards            = self.parser.training_cores()

    params = dict(
        self.config.as_dict(),
        model_name            = self.parser.model_name(),
        iterations_per_loop   = self.parser.iterations_per_loop(),
        model_dir             = self.parser.model_dir(),
        num_shards            = num_shards,
        num_examples_per_epoch= self.parser.examples_per_epoch(),
        strategy              = self.parser.strategy(),
        backbone_ckpt         = self.parser.backbone_checkpoint(),
        ckpt                  = self.parser.checkpoint(),
        val_json_file         = self.parser.val_json_file(),
        testdev_dir           = self.parser.testdev_dir(),
        profile               = self.parser.profile(),
        mode                  = self.parser.training_mode() )
    config_proto = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)

    if self.parser.strategy() != 'tpu':
      if self.parser.use_xla():
        config_proto.graph_options.optimizer_options.global_jit_level = (
            tf.OptimizerOptions.ON_1)
        config_proto.gpu_options.allow_growth = True

    model_dir               = self.parser.model_dir()
    model_fn_instance       = det_model_fn.get_model_fn(self.parser.model_name() )
    max_instances_per_image = self.config.max_instances_per_image
    print("=== max_instances_per_image {}".format(max_instances_per_image))
    
    if self.parser.eval_samples():
      self.eval_steps = int((self.parser.eval_samples() + self.parser.validation_batch_size() - 1) //
                       self.parser.validation_batch_size() )
    else:
      self.eval_steps = None
    print("=== self.eval_steps  {}".format(self.eval_steps))
    
    total_examples = int(self.config.num_epochs * self.parser.examples_per_epoch() )
    self.train_steps    = total_examples // self.parser.training_batch_size()
    
    logging.info(params)
    print("=== self.train_steps {}".format(self.train_steps))
    
    if not tf.io.gfile.exists(model_dir):
      tf.io.gfile.makedirs(model_dir)

    config_file = os.path.join(model_dir, 'config.yaml')
    if not tf.io.gfile.exists(config_file):
      print("=== Writing model_dir config.yaml {}  {}".format(config_file, str(self.config) ))
     
      tf.io.gfile.GFile(config_file, 'w').write(str(self.config))


    self.train_input_fn = dataloader.InputReader(
        self.parser.training_file_pattern(),
        is_training             = True,
        use_fake_data           = self.parser.use_fake_data(),
        max_instances_per_image = max_instances_per_image)
    
    self.eval_input_fn = dataloader.InputReader(
        self.parser.validation_file_pattern(),
        is_training             = False,
        use_fake_data           = self.parser.use_fake_data(),
        max_instances_per_image = max_instances_per_image)

    if self.parser.strategy() == 'tpu':
      tpu_config = tf.estimator.tpu.TPUConfig(
          self.parser.iterations_per_loop() if self.parser.strategy() == 'tpu' else 1,
          num_cores_per_replica       = num_cores_per_replica,
          input_partition_dims        = input_partition_dims,
          per_host_input_for_training = tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2)
      run_config = tf.estimator.tpu.RunConfig(
          cluster                = tpu_cluster_resolver,
          model_dir              = model_dir,
          log_step_count_steps   = self.parser.iterations_per_loop(),
          session_config         = config_proto,
          tpu_config             = tpu_config,
          save_checkpoints_steps = self.parser.save_checkpoints_steps(),
          tf_random_seed         = self.parser.tf_random_seed(),
      )
      # TPUEstimator can do both train and eval.
      self.train_estimator = tf.estimator.tpu.TPUEstimator(
          model_fn         = model_fn_instance,
          train_batch_size = self.parser.training_batch_size(),
          eval_batch_size  = self.parser.eval_batch_size(),
          config           = run_config,
          params           = params)
      self.eval_estimator = self.train_estimator
    else:
      strategy = None
      print("=== strategy is None")
      if self.parser.strategy() == 'gpus':
        strategy = tf.distribute.MirroredStrategy()
      run_config = tf.estimator.RunConfig(
          model_dir              = model_dir,
          train_distribute       = strategy,
          log_step_count_steps   = self.parser.iterations_per_loop(),
          session_config         = config_proto,
          save_checkpoints_steps = self.parser.save_checkpoints_steps(),
          tf_random_seed         = self.parser.tf_random_seed(),
      )

      def get_estimator(global_batch_size):
        params['num_shards'] = getattr(strategy, 'num_replicas_in_sync', 1)
        params['batch_size'] = global_batch_size // params['num_shards']
        return tf.estimator.Estimator(
            model_fn = model_fn_instance, config=run_config, params=params)

      # train and eval need different estimator due to different batch size.
      self.train_estimator = get_estimator(self.parser.training_batch_size() )
      self.eval_estimator  = get_estimator(self.parser.validation_batch_size() )
      print("=== created train_estimator")
      print("=== created eval_estimator")
      
    
  def train(self):
    ipaddress = self.parser.epoch_change_notifier_ipaddress()
    port      = self.parser.epoch_change_notifier_port()
    self.epoch_change_notifier = EpochChangeNotifier(ipaddress, port)    
    self.epoch_change_notifier.begin_training()

    # start train/eval flow.
    print("== training_mode {}".format(self.parser.training_mode() ))
    if self.parser.training_mode() == self.TRAIN:
      print("=== start train ")
      self.train_estimator.train(input_fn=self.train_input_fn, max_steps=self.train_steps)
      if self.parser.eval_after_train():
        self.eval_estimator.evaluate(input_fn=self.eval_input_fn, steps=self.eval_steps)

    elif self.parser.training_mode() == self.EVAL:
      print("=== run evaluation")
      # Run evaluation when there's a new checkpoint
      for ckpt in tf.train.checkpoints_iterator(
          self.parser.model_dir(),
          min_interval_secs = self.parser.min_eval_interval() ,
          timeout           = self.parser.eval_timeout() ):

        logging.info('Starting to evaluate.')
        try:
          eval_results = self.eval_estimator.evaluate(self.eval_input_fn, steps=self.eval_steps)
          # Terminate eval job when final checkpoint is reached.
          try:
            current_step = int(os.path.basename(ckpt).split('-')[1])
          except IndexError:
            logging.info('%s has no global step info: stop!', ckpt)
            break

          utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
          if current_step >= self.train_steps:
            logging.info('Eval finished step %d/%d', current_step, self.train_steps)
            break

        except tf.errors.NotFoundError:
          # Checkpoint might be not already deleted by the time eval finished.
          # We simply skip ssuch case.
          logging.info('Checkpoint %s no longer exists, skipping.', ckpt)

    elif self.parser.training_mode() == self.TRAIN_AND_EVAL:
      print("=== start train and eval")
      ckpt = tf.train.latest_checkpoint(self.parser.model_dir() )
      print("=== 'train_and_eval' {}".format(ckpt))
      try:
        step = int(os.path.basename(ckpt).split('-')[1])
        print("=== step {}".format(step))
        current_epoch = (
            step * self.parser.training_batch_size() // self.parser.examples_per_epoch() )
        logging.info('found ckpt at step %d (epoch %d)', step, current_epoch)
      except (IndexError, TypeError):
        logging.info('Folder %s has no ckpt with valid step.', self.parser.model_dir() )
        current_epoch = 0


      epochs_per_cycle = 1  # higher number has less graph construction overhead.
      #self.parser.epochs() = self.config.num_epochs      
      for e in range(current_epoch + 1, self.config.num_epochs + 1, epochs_per_cycle):
        #print("=== e {}".format(e))
        if self.parser.run_epoch_in_child_process():
          p = multiprocessing.Process(target=run_train_and_eval, args=(e,))
          p.start()
          p.join()
          if p.exitcode != 0:
            return p.exitcode
        else:
          tf.reset_default_graph()
          breaking_loop = self.run_train_and_eval(e)
          if breaking_loop == True:
            print("=== Breaking the train_and_eval loop")
            break

    else:
      logging.info('Invalid mode: %s', self.parser.training_mode())


  def run_train_and_eval(self, e):
        print('\n==> Starting training, epoch: %d.' % e)
        max_steps = e * self.parser.examples_per_epoch() // self.parser.training_batch_size()
        print("=== examples_per_epoch  {}".format(self.parser.examples_per_epoch()))
        print("=== training_batch_size {}".format(self.parser.training_batch_size()))
        print("=== max_steps           {}".format(max_steps))
        self.train_estimator.train(
            input_fn  = self.train_input_fn,
            max_steps = max_steps )
        print('\n==> Starting evaluation, epoch: %d.' % e)
        eval_results = self.eval_estimator.evaluate(input_fn=self.eval_input_fn, steps=self.eval_steps)
        print("=== eval_results {}".format(eval_results))
        
        map = eval_results['AP50']
        loss = eval_results['loss']

        self.epoch_change_notifier.epoch_end(e-1, loss, map)
        
        self.evaluation_results_writer.write(e-1, eval_results)
        self.training_losses_writer.write(e-1, eval_results)
                
        ckpt = tf.train.latest_checkpoint(self.parser.model_dir() )
        utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)

        breaking_loop = False
        if self.early_stopping != None:
          mAP           = eval_results['AP50']
          breaking_loop = self.early_stopping.validate(mAP)
        return breaking_loop


if __name__ == '__main__':
  try:
    train_config = ""
    if len(sys.argv) >=2:
      train_config = sys.argv[1]
      if not os.path.exists(train_config):
        print("Usage: python EfficientDetFinetuningModel.py  ./projects/BloodCells/configs/train.config")
        raise Exception("Not found " + train_config)
        
    model = EfficientDetFinetuningModel(train_config)
    model.train()
    
  except Exception as ex:
    traceback.print_exc()
    
