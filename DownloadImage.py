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
# DownloadImage.py

import sys
import os
import time
import traceback
import numpy as np
import tarfile
import shutil


def download_image_file(img_file):

  try:   
    path = os.path.join(os.getcwd(), "images")
    os.makedirs(path, exist_ok=True)
    local_image_path = os.path.join(path, img_file)
    if os.path.exists(local_image_path) != True:

         url = 'https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png'
         print("Downloading a file {}".format(url))

         img_file = tf.keras.utils.get_file(img_file, url)
         shutil.move(img_file, local_image_path)

         print("You have downloaded {}".format(local_image_path))
    else:
         print("Found a downloaded file {}".format(local_image_path))

    return local_image_path

  except Exception as ex:
    traceback.print_exc()


if __name__=="__main__":
         
  try:
      img_file="img.png"

      download_image_file(img_file)

  except Exception as ex:
    traceback.print_exc()

