#******************************************************************************
#
#  Copyright (c) 2020-2021 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
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
#
#******************************************************************************
# 
# EfficientDet 
# EfficientDetector.py

# 2020/06/17:
# Updated EfficientDetector class
# Added detect_all method to DetectionTransformer class.
#   def detect_all(self, input_image_dir, output_image_dir):
#
# This method detects objectes in each image in input_image_dir, and saves the detected image 
# to output_image_dir.
#
# 2020/07/23 
#  Updated inference.py and visualize/vis_utils.py to get detected_objects information
#
# 2021/02/10 
# Updated to use detect.config file.
    
import sys
import os
import glob
import time
import traceback
import matplotlib
matplotlib.use('Agg')  # Set headless-friendly backend.
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import PIL

from PIL import Image
import tensorflow.compat.v1 as tf
import inference2 

#2021/2/10 Updated FiltersParser class.
from FiltersParser import FiltersParser

from DetectConfigParser import DetectConfigParser

class EfficientDetector:
  #Constructor
  #
  def __init__(self, detect_config):
    self.parser            = DetectConfigParser(detect_config)
    
    self.min_score_thresh  = self.parser.threshold()  
    self.max_boxes_to_draw = self.parser.max_boxes() 
    self.line_thickness    = self.parser.line_thickness()

    self.model             = self.parser.model_name()
    self.output_dir        = self.parser.output_dir()
    self.ckpt_path         = self.parser.checkpoint_dir()
    if not os.path.exists(self.output_dir):
        os.makedirs(self.output_dir)

    tf.enable_v2_tensorshape()
    tf.disable_eager_execution()


  def detect_all(self, input_image_dir, filters=None):
    if not os.path.exists(input_image_dir):
        raise Exception("Not found input_image_dir {}".format(input_image_dir))
    
    image_list = []

    if os.path.isdir(input_image_dir):
      image_list.extend(glob.glob(os.path.join(input_image_dir, "*.png")) )
      image_list.extend(glob.glob(os.path.join(input_image_dir, "*.jpg")) )

    print("image_list {}".format(image_list) )
        
    for image_filename in image_list:
        #image_filename will take images/foo.png
        image_file_path = os.path.abspath(image_filename)
        
        print("filename {}".format(image_file_path))
        
        self.detect(image_file_path, filters)


  def detect(self, image_file, filters=None ):
    if not os.path.exists(image_file):
        raise Exception("Not found image_file " + image_file) 

    filtersParser = FiltersParser()
    filters = filtersParser.parse(filters)
    print("=== {}".format(filters))
      
    tf.reset_default_graph()
    
    image_size  = max(PIL.Image.open(image_file).size)
    print("ImageSize {}".format(image_size))

    self.model_params= {"image_size": image_size}

    self.driver = inference2.InferenceDriver2(self.model, 
                            self.ckpt_path, 
                            model_params=self.model_params) 
       
    print("Start inference")
    start = time.time()

    predictions = self.driver.inference(filters, 
               image_file,
               self.output_dir,
               min_score_thresh  = self.min_score_thresh,
               max_boxes_to_draw = self.max_boxes_to_draw,
               line_thickness    = self.line_thickness)
    print("Done inference")
    elapsed_time = time.time() - start
    print("Elapsed_time:{0}".format(elapsed_time) + "[sec]")



########################
#
if __name__=="__main__":

  try:
     if len(sys.argv) < 3:
        #python EfficientDetector.py images/img.png ./projects/coco/configs/detect.config [car,person]
        raise Exception("Usage: {} image_file_or_dir detect.config filters".format(sys.argv[0]))
        
     input_image_file = None
     input_image_dir  = None
     detect_config    = None
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
         
     if len(sys.argv) == 4:
       filters = sys.argv[3]

     detector = EfficientDetector(detect_config)

     if input_image_dir is not None:
         detector.detect_all(input_image_dir, filters )
       
     if input_image_file is not None:
         detector.detect(input_image_file, filters)

  
  except Exception as ex:
    traceback.print_exc()

