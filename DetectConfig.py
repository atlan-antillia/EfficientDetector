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

#DetectConfig.py

class DetectConfig: 

  DETECTION         = "detection"
  
  MODEL_NAME        = "name"
  #efficientdet-d0
  
  LOG_DIR           = "logdir"

  LABEL_MAP_PBTXT   = "label_map_pbtxt"  

  
  TRACE_FILENAME    = "trace_filename"
  
  THREADS           = "threads"
  #'Number of threads.'
  
  
  DELETE_LOGDIR     = "delete_logdir"
  
  
  BATCH_SIZE        = "batch_size"
  #'Batch size for inference.'
  
  CHECKPOINT_DIR        = "checkpoint_dir"
  #'checkpoint dir used for eval.'
  
  SAVEDMODEL_DIR        = "savedmodel_dir"
  #'checkpoint dir used for eval.'
  
  HPARAMS            = "hparams"
  #'Comma separated k=v pairs of hyperparameters or a module'
  #  ' containing attributes to use as hyperparameters.')
  
  OUTPUT_DIR         = "output_dir"
  

  VISUALIZATION     = "visualization"
  LINE_THICKNESS    = "line_thickness"
  
  MAX_BOXES         = "max_boxes"
  #'Max number of boxes to draw.
  
  THRESHOLD         = "threshold"
  #'Score threshold to show box.'
  
  NMS_METHOD        = "nms_method"
  #'nms method, hard or gaussian.'
  
  def __init__(self):
    pass
    
  
  def parse_if_possible(self, val):
   val = str(val)
   if val == "None":
     return None
   if val == "False":
     return False
   if val == "True":
     return True
   return val
   

