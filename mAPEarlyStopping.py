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

#2021/02/10 
#mAParlyStopping.py

import os
import sys

class mAPEarlyStopping():
  def __init__(self, patience=5, verbose=0):
      self._step    = 0
      self._mAP     = 0.0
      self.patience = patience
      self.verbose  = verbose

  def validate(self, mAP):
       
      if self._mAP > mAP:
          #Not increasing mAP
          self._step += 1
          print("=== mAPEarlyStopping step:{} prev mAP:{} > new mAP: {}".format(self._step, self._mAP, mAP))
          
          if self._step > self.patience:
              if self.verbose:
                  print('=== mAPEarlyStopping is validated')
              return True
      else:
          # self._mAP <= mAP
          print("=== mAPEarlyStopping step:{} prev mAP:{} <= new mAP:{}".format(self._step, self._mAP, mAP))

          self._step = 0
          self._mAP = mAP

      return False

