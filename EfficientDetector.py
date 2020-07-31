#******************************************************************************
#
#  Copyright (c) 2020 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
#import inference 
import inference2 

from FiltersParser import FiltersParser

class EfficientDetector:

  def __init__(self, model="efficientdet-d0"):
      self.min_score_thresh  = 0.4  
      self.max_boxes_to_draw = 200  
      self.line_thickness    = 2
 
      self.MODEL       = model #"efficientdet-d0"
      self.ckpt_path   = os.path.join(os.getcwd(), self.MODEL)


  #2020/06/17
  # Detect objectes in each image in input_image_dir, and save the detected image 
  # to output_image_dir.
    
  def detect_all(self, filters, input_image_dir, output_image_dir):
      if not os.path.exists(input_image_dir):
          raise Exception("Not found input_image_dir {}".format(input_image_dir))

      output_image_dir = os.path.join(os.getcwd(), output_image_dir)
      if not os.path.exists(output_image_dir):
          os.makedirs(output_image_dir)
      
      image_list = []

      if os.path.isdir(input_image_dir):
        image_list.extend(glob.glob(os.path.join(input_image_dir, "*.png")) )
        image_list.extend(glob.glob(os.path.join(input_image_dir, "*.jpg")) )

      print("image_list {}".format(image_list) )
          
      for image_filename in image_list:
          #image_filename will take images/foo.png
          image_file_path = os.path.abspath(image_filename)
          
          print("filename {}".format(image_file_path))
          
          predictions = self.detect(filters, image_file_path, output_image_dir)


  def detect(self, filters, input_image_filepath, output_image_dir="detected"):
      if not os.path.exists(input_image_filepath):
          raise Exception("Not found image_file {}".format(input_image_filepath)) 
      
      output_image_dir = os.path.join(os.getcwd(), output_image_dir)
      if not os.path.exists(output_image_dir):
          os.makedirs(output_image_dir)
          
      tf.reset_default_graph()
      
      image_size  = max(PIL.Image.open(input_image_filepath).size)
      print("ImageSize {}".format(image_size))

      self.model_params= {"image_size": image_size}

      self.driver = inference2.InferenceDriver2(self.MODEL, 
                                              self.ckpt_path, 
                                 model_params=self.model_params) 
         
      print("Start inference")
      start = time.time()

      predictions = self.driver.inference(filters, 
                 input_image_filepath,
                 output_image_dir,
                 min_score_thresh  = self.min_score_thresh,
                 max_boxes_to_draw = self.max_boxes_to_draw,
                 line_thickness    = self.line_thickness)
      print("Done inference")
      elapsed_time = time.time() - start
      print("Elapsed_time:{0}".format(elapsed_time) + "[sec]")
      return predictions



########################
#
if __name__=="__main__":

  try:
     if len(sys.argv) < 3:
        raise Exception("Usage: {} input_image_file output_image_dir filters".format(sys.argv[0]))
        
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
       output_image_dir = sys.argv[2]
       if not os.path.exists(output_image_dir):
         os.makedirs(output_image_dir)
     if len(sys.argv) == 4:
       str_filters = sys.argv[3]
       filtersParser = FiltersParser(str_filters)
       filters = filtersParser.get_filters()
       
     if input_image_file is not None and output_image_dir is not None:

         detector = EfficientDetector()
         detector.detect(filters, input_image_file, output_image_dir)

     if input_image_dir is not None and output_image_dir is not None:
         detector       = EfficientDetector()
         detector.detect_all(filters, input_image_dir, output_image_dir)
  
  except Exception as ex:
    traceback.print_exc()

