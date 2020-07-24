# Lint as: python3
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
"""A simple example on how to use keras model for inference."""
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

import hparams_config
import inference
import utils
from keras import efficientdet_keras

flags.DEFINE_string('image_path', None, 'Location of test image.')
flags.DEFINE_string('output_dir', None, 'Directory of annotated output images.')
flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
flags.DEFINE_bool('debug', False, 'If true, run function in eager for debug.')
FLAGS = flags.FLAGS


def main(_):
  # pylint: disable=line-too-long
  # Prepare images and checkpoints: please run these commands in shell.
  # !mkdir tmp
  # !wget https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png -O tmp/img.png
  # !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz -O tmp/efficientdet-d0.tar.gz
  # !tar zxf tmp/efficientdet-d0.tar.gz -C tmp
  imgs = [np.array(Image.open(FLAGS.image_path))]
  # Create model config.
  config = hparams_config.get_efficientdet_config('efficientdet-d0')
  config.is_training_bn = False
  config.image_size = '1920x1280'
  config.nms_configs.score_thresh = 0.4
  config.nms_configs.max_output_size = 100
  config.override(FLAGS.hparams)

  # Use 'mixed_float16' if running on GPUs.
  policy = tf.keras.mixed_precision.experimental.Policy('float32')
  tf.keras.mixed_precision.experimental.set_policy(policy)
  tf.config.experimental_run_functions_eagerly(FLAGS.debug)

  # Create and run the model.
  model = efficientdet_keras.EfficientDetModel(config=config)
  height, width = utils.parse_image_size(config['image_size'])
  model.build((1, height, width, 3))
  model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))
  model.summary()

  @tf.function
  def f(imgs):
    return model(imgs, training=False, post_mode='global')

  boxes, scores, classes, valid_len = f(imgs)

  # Visualize results.
  for i, img in enumerate(imgs):
    length = valid_len[i]
    img = inference.visualize_image(
        img,
        boxes[i].numpy()[:length],
        classes[i].numpy().astype(np.int)[:length],
        scores[i].numpy()[:length],
        min_score_thresh=config.nms_configs.score_thresh,
        max_boxes_to_draw=config.nms_configs.max_output_size)
    output_image_path = os.path.join(FLAGS.output_dir, str(i) + '.jpg')
    Image.fromarray(img).save(output_image_path)
    print('writing annotated image to ', output_image_path)


if __name__ == '__main__':
  flags.mark_flag_as_required('image_path')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('checkpoint')
  logging.set_verbosity(logging.WARNING)
  app.run(main)