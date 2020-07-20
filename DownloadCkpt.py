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

# EfficientDet 
# DownloadCkpt.py

import sys
import os
import time
import traceback
import numpy as np
import tarfile
import shutil
import tensorflow as tf


def download_checkpoint_file():

  try:
      #Download checkpoint file
      url = "https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz"
      folder = "efficientdet-d0"
      tar_file = "efficientdet-d0.tar.gz"

      if os.path.exists(folder) != True:
          print("Try download {}".format(url))

          tar_file = tf.keras.utils.get_file(tar_file, url)
          print("You have downloaded {}".format(tar_file))

          with tarfile.open(tar_file, "r:gz") as tar:
             tar.extractall()
      else:
          print("OK, you have the weight file {}!".format(tar_file))
       
  except Exception as ex:
    traceback.print_exc()


if __name__=="__main__":
         
  try:
      MODEL = "efficientdet-d0"
 
      ckpt_path = os.path.join(os.getcwd(), MODEL);

      download_checkpoint_file()

  except Exception as ex:
    traceback.print_exc()

