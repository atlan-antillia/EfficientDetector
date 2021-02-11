# Copyright 2020-2021 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# TrainConfigParser.py
#
import os
import sys
import glob
import json
from collections import OrderedDict
import pprint
import configparser 
import traceback

from TrainConfig import TrainConfig

#Comments were taken from main.py

class TrainConfigParser(TrainConfig):

  # Constructor
  # 
  def __init__(self, train_config_path):
    print("==== TrainConfigParser {}".format(train_config_path))
    if not os.path.exists(train_config_path):
      raise Exception("Not found train_config_path {}".format(train_config_path))

    try:
      self.parse(train_config_path)
    except Exception as ex:
      traceback.print_exc()


  def parse(self, train_config_path):
    self.config = configparser.ConfigParser()
    self.config.read(train_config_path)
    self.dump_all()


  def project_name(self):
    try:
      return self.config[self.PROJECT][self.NAME]
    except:
      return None
      
      
  def owner(self):
    try:
      return self.config[self.MODEL][self.OWNER]
    except:
      return None
      
  def dataset(self):
    try:
      return self.config[self.MODEL][self.DATASET]
    except:
      return None 
      
  """
  'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
  """
  def tpu(self):
    rc = False
    try:
      val = self.config[self.HARDWARE][self.TPU]
      rc = self.parse_if_possible(val)
    except:
      pass
    return rc

  """
  'GCE zone where the Cloud TPU is located in. If not specified, we '
  'will attempt to automatically detect the GCE project from metadata.')
  """    
  def tpu_zone(self):
    rc = False
    try:
      val = self.config[self.HARDWARE][self.TPU_ZONE]
      rc = self.parse_if_possible(val)
    except:
      pass
    return rc


  """
  'Project name for the Cloud TPU-enabled project. If not specified, we '
  'will attempt to automatically detect the GCE project from metadata.'

  """
  def gcp_project(self):
    try:
      val = self.config[self.HARDWARE][self.GCP_PROJECT]
      return self.parse_if_possible(val)
    except:
      return None
  
  """
  'Training: gpus for multi-gpu, if None, use TF default.'
  """
  def strategy(self):
    try:
      val = self.config[self.HARDWARE][self.STRATEGY]
      return self.parse_if_possible(val)
    except:
      return None
      
  """
  'Use XLA even if strategy is not tpu. If strategy is tpu, always use XLA, '
  'and this flag has no effect.'
  """
  def use_xla(self):
    rc = False
    try:
      val = self.config[self.HARDWARE][self.USE_XLA]
      rc = self.parse_if_possible(val)      
    except:
      pass
    return rc


  """
  'Model name.'
  """
  def model_name(self):
    try:
      return self.config[self.MODEL][self.NAME]
    except:
      return "efficientdet-d1"
      
  """
  'Location of model_dir'

  """ 
  def model_dir(self):
    try:
      return self.config[self.MODEL][self.MODEL_DIR]
    except:
      return None
  
  """
  'Start training from this EfficientDet checkpoint.'
  """    
  def checkpoint(self):
    try:
      val = self.config[self.MODEL][self.CHECKPOINT]
      return self.parse_if_possible(val)
    except:
      return None

  """
  'Start training from this EfficientDet checkpoint.'
  """    
  def backbone_checkpoint(self):
    try:
      val = self.config[self.MODEL][self.BACKBONE_CHECKPOINT]
      return self.parse_if_possible(val)
    except:
      return None

  def profile(self):
    try:
      val = self.config[self.MODEL][self.PROFILE]
      return self.parse_if_possible(val)
    except:
      return None

  def training_mode(self):
    try:
      return self.config[self.TRAINING][self.MODE]
    except:
      return "train"

  def training_csv_path(self):
    try:
      return self.config[self.TRAINING][self.TRAINING_CSV_PATH]
    except:
      return "./training.csv"

  # 'global training batch size'

  def training_batch_size(self):
    try:
      return int(self.config[self.TRAINING][self.BATCH_SIZE])
    except:
      return 1
      
  ###
  def epochs(self):
    try:
      return int(self.config[self.TRAINING][self.EPOCHS])
    except:
      return 1

  """
  'This option helps to rectify CPU memory leak. If True, every epoch is '
  'run in a separate process for train and eval and memory will be cleared.'
  'Drawback: need to kill 2 processes if trainining needs to be interrupted.'
  """
  def run_epoch_in_child_process(self):
    try:
      val = self.config[self.HARDWARE][self.RUN_EPOCH_IN_CHILD_PROCESS]
      return self.parse_if_possible(val)      
    except:
      return False
    
    
  def save_checkpoints_steps(self):
    try:
      return int(self.config[self.TRAINING][self.SAVE_CHECKPOINTS_STEPS])
    except:
      return 1
    

  def training_file_pattern(self):
    try:
      return self.config[self.TRAINING][self.FILE_PATTERN]
    except:
      return None
  
  # Comma separated k=v pairs of hyperparameters or a module
  # containing attributes to use as hyperparameters.
  def hparams(self):
    try:
      val = self.config[self.TRAINING][self.HPARAMS]
      return self.parse_if_possible(val)
    except:
      pass
    return None

  # 'Number of examples in one epoch'
  def examples_per_epoch(self):
    try:
      return int(self.config[self.TRAINING][self.EXAMPLES_PER_EPOCH])
    except:
      return 1000
 
  """
  'Number of TPU cores for training'
  """
  def training_cores(self):
    try:
      return int(self.config[self.TRAINING][self.CORES])
    except:
      pass
    return self.INVALID
    
  """
  'Use spatial partition.'
  """
  def use_spatial_partition(self):
    rc = False
    try:
      val = self.config[self.TRAINING][self.USE_SPATIAL_PARTITION]
      rc = self.parse_if_possible(val)
    except:
      pass
    return rc
    
  def use_fake_data(self):
    rc = False
    try:
      val = self.config[self.TRAINING][self.USE_FAKE_DATA]
      rc = self.parse_if_possible(val)
    except:
      pass
    return rc

  def cores_per_replica(self):
    try:
      return int(self.config[self.TRAINING][self.CORES_PER_REPLICA])
    except:
      return self.INVALID
      
     
  def input_partition_dims(self):
    try:
      return list(self.config[self.TRAINING][self.INPUT_PARTION_DIMS])
    except:
      return [1, 2, 1, 1]
  
  """
  'Sets the TF graph seed for deterministic execution'
  ' across runs (for debugging).'
  """
  def tf_random_seed(self):
    try:
      val = self.config[self.TRAINING][self.TF_RANDOM_SEED]
      return self.symbolize_if_possible(val)
    except:
      return False

  def training_losses_file(self):
    try:
      return self.config[self.TRAINING][self.TRAINING_LOSSES_FILE]
    except:
      pass
    return None

  ####validation  
  """
  'Glob for evaluation tfrecords (e.g., COCO val2017 set)'
  """
  def validation_file_pattern(self):
    try:
      return self.config[self.VALIDATION][self.FILE_PATTERN]
    except:
      return None
    
  """
  'global evaluation batch size'
  """
  def validation_batch_size(self):
    try:
      return int(self.config[self.VALIDATION][self.BATCH_SIZE])
    except:
      return 1
    
  """
  'Number of samples for eval.'
  """
  def eval_samples(self):
    try:
      return int(self.config[self.VALIDATION][self.EVAL_SAMPLES])
    except:
      return 1000
      
  """
  'Number of iterations per TPU training loop'
  """
  def iterations_per_loop(self):
    try:
      return int(self.config[self.VALIDATION][self.ITERATION_PER_LOOP])
    except:
      return 100
      
  """
  'COCO validation JSON containing golden bounding boxes. If None, use the '
  'ground truth from the dataloader. Ignored if testdev_dir is not None.')

  """
  def val_json_file(self):
    try:
      val = self.config[self.VALIDATION][self.VAL_JSON_FILE]
      return self.parse_if_possible(val)
    except:
      return None
      

  def testdev_dir(self):
    try:
      val = self.config[self.VALIDATION][self.TESTDEV_DIR]
      return self.parse_if_possible(val)
    except:
      return None
      

  """
  'Run one eval after the training finishes.'
  """
  def eval_after_train(self):
    try:
      val = str(self.config[self.VALIDATION][self.EVAL_AFTER_TRAIN])
      return self.parse_if_possible(val)
    except:
      return False
    
  """
  'Minimum seconds between evaluations.'
  """
  def min_eval_interval(self):
    try:
      return int(self.config[self.VALIDATION][self.MIN_EVAL_INTERVAL])
    except:
      return None
      
  """
  'Maximum seconds between checkpoints before evaluation terminates.'
  """    
  def eval_timeout(self):
    try:
      return int(self.config[self.VALIDATION][self.EVAL_TIMEOUT])
    except:
      return None


  def evaluation_results_file(self):
    try:
      return self.config[self.VALIDATION][self.EVALUATION_RESULTS_FILE]
    except:
      return None
      

  def early_stopping_patience(self):
    MAX_PATIENCE = 50
    try:   
      val =int(self.config[self.EARLY_STOPPING][self.PATIENCE])
      if not (val > 0 and val < MAX_PATIENCE):
        val = self.INVALID
      return val
    except:
      return self.INVALID

  
  def epoch_change_notifier_enabled(self):
    try:
      val = self.config[self.EPOCH_CHANGE_NOTIFIER][self.ENABLED]
      rc = self.symobolize_if_possible(val)
      return rc
   
    except:
      return False
      
  def epoch_change_notifier_ipaddress(self):
    try:
      return self.config[self.EPOCH_CHANGE_NOTIFIER][self.IPADDRESS]
    except:
      return None
    
  def epoch_change_notifier_port(self):
    try:
      return int(self.config[self.EPOCH_CHANGE_NOTIFIER][self.PORT])   
    except:
      return 9999
    
  
  #--------------------------------------------------------------
  def dump_all(self):
  
    print("project_name           {}".format(self.project_name() ))
    print("owner                  {}".format(self.owner() ))
    
    print("dataset                {}".format(self.dataset() ))

    print("tpu                    {}".format(self.tpu() ))

    print("tpu_zone               {}".format(self.tpu_zone() ))
  
    print("gcp_project            {}".format(self.gcp_project() ))

    print("strategy               {}".format(self.strategy() ))

    print("use_xla                {}".format(self.use_xla() ))

    print("model_name             {}".format(self.model_name() ))

    print("model_dir              {}".format(self.model_dir() ))

    print("checkpoint             {}".format(self.checkpoint() ))

    print("hparams                {}".format(self.hparams() ))

    print("training_mode          {}".format(self.training_mode() ))

    print("training_batch_size    {}".format(self.training_batch_size() ))

    print("epochs                 {}".format(self.epochs() ))

    print("save_checkpoints_steps {}".format(self.save_checkpoints_steps() ))

    print("training_file_pattern  {}".format(self.training_file_pattern() ))

    print("examples_per_epoch     {}".format(self.examples_per_epoch() ))

    print("training_cores         {}".format(self.training_cores() ))

    print("use_spatial_partition  {}".format(self.use_spatial_partition() ))

    print("tf_random_seed         {}".format(self.tf_random_seed() ))


    ####validation
    print("validation_file_pattern{}".format(self.validation_file_pattern() ))
  
    print("validation_batch_size  {}".format(self.validation_batch_size() ))

    print("eval_samples           {}".format(self.eval_samples() ))

    print("iterations_per_loop    {}".format(self.iterations_per_loop() ))

    print("val_json_file          {}".format(self.val_json_file() ))

    print("eval_after_train       {}".format(self.eval_after_train() ))
      
    print("min_eval_interval      {}".format(self.min_eval_interval() ))

    print("eval_timeout           {}".format(self.eval_timeout() ))

    print("evaluation_results_file {}".format(self.evaluation_results_file() ))

    print("early_stopping_patience {}".format(self.early_stopping_patience() ))

    print("                        {}".format(self.epoch_change_notifier_enabled() ))
      
    print("                        {}".format(self.epoch_change_notifier_ipaddress() ))
    print("                        {}".format(self.epoch_change_notifier_port() ))
  
##
##
##
if __name__ == "__main__":
  config_file = ""

  try:

    if len(sys.argv) >=2:
       config_file = sys.argv[1]
    if not os.path.exists(config_file):
        raise Exception("Not found {}".format(config_file))

    print("{}".format(config_file))
    train_config = TrainConfigParser(config_file)
    
    train_config.dump_all()
        
  except Exception as ex:
    traceback.print_exc()
    
     