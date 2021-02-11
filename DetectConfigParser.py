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
# DetectConfigParser.py
#
import os
import sys
import glob
import json
from collections import OrderedDict
import pprint
import configparser 
import traceback

from DetectConfig import DetectConfig

class DetectConfigParser(DetectConfig):

  def __init__(self, detect_config):
    self.detect_config = detect_config
    if not os.path.exists(self.detect_config):
      raise Exception("Not found " + self.detect_config)
      
    try:
      self.parse(self.detect_config)
      
    except Exception as ex:
      print(ex)
      
  def parse(self, detect_config):
    self.config = configparser.ConfigParser()
    self.config.read(detect_config)
    
    self.dump_all()

  def model_name(self):
    try:
      return self.config[self.DETECTION][self.MODEL_NAME]
    except:
      return "efficientdet-d0"
      
  def log_dir(self):
    try:
      return self.config[self.DETECTION][self.LOG_DIR]
    except:
      return None

  def label_map_pbtxt(self):
    try:
      return self.config[self.DETECTION][self.LABEL_MAP_PBTXT]
    except:
      return None

    
  def delete_logdir(self):
    try:
      val = self.config[self.DETECTION][self.DELETE_LOGDIR]
      return self.symbolize_if_possible(val)
    except:
      return False
    

  def batch_size(self):
    try:
      return int(self.config[self.DETECTION][self.BATCH_SIZE])
    except:
      return 1

  def checkpoint_dir(self):
    try:
      return self.config[self.DETECTION][self.CHECKPOINT_DIR]
    except:
      return None
       
  def savedmodel_dir(self):
    try:
      return self.config[self.DETECTION][self.SAVEDMODEL_DIR]
    except:
      return None

  def hparams(self):
    try:
      return self.config[self.DETECTION][self.HPARAMS]
    except:
      return None
  
  def output_dir(self):
    try:
      return self.config[self.DETECTION][self.OUTPUT_DIR]
    except Exception as ex:
      return None
  
  
  # VISUALIZATION
  def line_thickness(self):
    try:
      return int(self.config[self.VISUALIZATION][self.OUTPUT_VIDEO])
    except:
      return 2
  
  def max_boxes(self):
    try:
      return int(self.config[self.VISUALIZATION][self.MAX_BOXES])
    except:
      return 2
  
  def threshold(self):
    try:
      return float(self.config[self.VISUALIZATION][self.THRESHOLD])
    except:
      return 0.4

  def nms_method(self):
    try:
      return self.config[self.VISUALIZATION][self.NMS_METHOD]
    except:
      return "hard"
  
  def dump_all(self):
      
    print("model_name           {}".format(self.model_name() ))

    print("log_dir              {}".format(self.log_dir() ))

    print("label_map_pbtxt      {}".format(self.label_map_pbtxt() ))

    print("delete_logdir        {}".format(self.delete_logdir() ))

    print("batch_size           {}".format(self.batch_size() ))

    print("checkpoint_dir       {}".format(self.checkpoint_dir() ))
    
    print("savedmodel_dir       {}".format(self.savedmodel_dir() ))

    print("hparams              {}".format(self.hparams() ))
    
    print("output_dir           {}".format(self.output_dir() ))

    print("line_thickness       {}".format(self.line_thickness() ))
 
    print("max_boxes            {}".format(self.max_boxes() ))

    print("threshold            {}".format(self.threshold() ))

    print("nms_method           {}".format(self.nms_method() ))


if __name__ == "__main__":
  try:
    detect_config = "./projects/BloodCells/configs/detect.config"
    parser = InspectConfigParser(detect_config)
    
  except Exception as ex:
    traceback.print_exc()
    
