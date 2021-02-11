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


#2021/02/10 Copyright 2021 Toshiyuki Arai antillia.com
#
#This is a very simpilied object detector which derived from ModelInspector class in model_inspect.py

import os
import sys
import glob
import time
from typing import Text, Tuple, List
import traceback

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

import hparams_config
import utils
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import

from DetectConfigParser import DetectConfigParser
import inference2 
import inference

from FiltersParser import FiltersParser
from DetectConfigParser import DetectConfigParser
from LabelMapReader import LabelMapReader


class EfficientDetObjectDetector(object):
  """This is a simple object detector class based on model_inspect.py."""

  def __init__(self, detect_config):
    print("=== EfficientDetObjectDetector")

    self.parser                 = DetectConfigParser(detect_config)
    self.model_name             = self.parser.model_name()
    self.logdir                 = self.parser.log_dir()
    self.delete_logdir          = self.parser.delete_logdir()
    self.checkpoint             = self.parser.checkpoint_dir()
    self.savedmodel_dir         = self.parser.savedmodel_dir()
    
    self.batch_size             = self.parser.batch_size()
    self.hparams                = self.parser.hparams()
    self.output_dir             = self.parser.output_dir()
    self.use_xla                = True
    
    if not os.path.exists(self.output_dir):
      print("=== Creating output_dir {}".format(self.output_dir))
      os.makedirs(self.output_dir)
    
    logging.set_verbosity(logging.WARNING)
    tf.enable_v2_tensorshape()
    tf.disable_eager_execution()

    if tf.io.gfile.exists(self.logdir) and self.delete_logdir:
      logging.info('Deleting log dir ...')
      tf.io.gfile.rmtree(self.logdir)

    self.build_model_config()
    
    self.label_map()


  def build_model_config(self):
  
    model_config = hparams_config.get_detection_config(self.model_name)   
    model_config.override(self.hparams)  # Add custom overrides
    model_config.is_training_bn = False
    model_config.image_size = utils.parse_image_size(model_config.image_size)

    # If batch size is 0, then build a graph with dynamic batch size.
    #self.batch_size   = self.batch_size or None
    #self.labels_shape = [self.batch_size, model_config.num_classes]

    # A hack to make flag consistent with nms configs.
    model_config.nms_configs.score_thresh = self.parser.threshold()
    model_config.nms_configs.method       = self.parser.nms_method()    
    #model_config.nms_configs.max_output_size = kwargs['max_output_size']

    height, width = model_config.image_size
    if model_config.data_format == 'channels_first':
      self.inputs_shape = [batch_size, 3, height, width]
    else:
      self.inputs_shape = [self.batch_size, height, width, 3]
    #print("=== model_config {}".format(model_config))
    
    self.model_config = model_config
    
    self.model_config_dict = self.model_config.as_dict()


  def label_map(self):
    try:
      label_map_file  = self.parser.label_map_pbtxt()
      print("=== label_map_file {}".format(label_map_file))
      reader = LabelMapReader()
      self.label_map_dict, self.classes = reader.read(label_map_file)
      print("=== label_map_dict {}".format(self.label_map_dict))
      print("=== classes        {}".format(self.classes))
    except Exception as ex:
      traceback.print_exc()


  #2021/02/10 Modified not to call self.detect.
  def detect_all(self, image_dir, filters=None):
    filtersParser = FiltersParser(self.classes)
    filters = filtersParser.parse(filters)
    print("=== detect_all filters {}".format(filters))
    
    if not os.path.exists(input_image_dir):
        raise Exception("Not found input_image_dir {}".format(input_image_dir))

    driver = inference2.InferenceDriver2(self.model_name, 
                                       self.checkpoint,
                                       model_params=self.model_config_dict)
    #2021/02/10 We limited image_file_pattern jpg files only
    image_file_pattern = os.path.join(image_dir, "*.jpg")

    driver.inference(filters, image_file_pattern,
               self.output_dir,
               min_score_thresh  = self.parser.threshold(),
               max_boxes_to_draw = self.parser.max_boxes(),
               line_thickness    = self.parser.line_thickness() )



  def detect(self, image_file, filters=None):
    print("=== detect image_file:{}".format(image_file))
    filtersParser = FiltersParser(self.classes)
    filters = filtersParser.parse(filters)
    
    driver = inference2.InferenceDriver2(self.model_name, 
                                       self.checkpoint,
                                       model_params=self.model_config_dict)


    driver.inference(filters, image_file, self.output_dir,
               min_score_thresh  = self.parser.threshold(),
               max_boxes_to_draw = self.parser.max_boxes(),
               line_thickness    = self.parser.line_thickness() )
    

if __name__ == '__main__':
  detect_config = None
  
  #image_file_path = "./projects/BloodCells/test/BloodCell_1.jpg"
  #image_file_dir  = "./projects/BloodCells/test/"
  
  try:
     if len(sys.argv) < 3:
        raise Exception("Usage: {} image_file_or_dir detect.config [filters]".format(sys.argv[0]))
        
     input_image_file = None
     input_image_dir  = None
     output_image_dir = None
     filters          = None  # classnames_list something like this "[person,car]"
     
     if len(sys.argv) >= 2:
       input = sys.argv[1]
       if not os.path.exists(input):
         raise Exception("Not found input {}".format(input))
       if os.path.isfile(input):
         input_image_file = input
       else:
         input_image_dir  = input

     if len(sys.argv) >= 3:
       detect_config = sys.argv[2]
       if not os.path.exists(detect_config):
         raise Exception("Not found " + detect_config)

     if len(sys.argv)>= 4:
       filters = sys.argv[3]

     detector       = EfficientDetObjectDetector(detect_config)

     if input_image_dir is not None:
         detector.detect_all(input_image_dir, filters)
       
     if input_image_file is not None:
         detector.detect(input_image_file,    filters)

  except Exception as ex:
    traceback.print_exc()


