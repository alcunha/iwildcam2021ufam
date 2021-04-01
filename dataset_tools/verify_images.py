# Copyright 2021 Fagner Cunha
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

import os
import json

from absl import app
from absl import flags

import pandas as pd
import tensorflow as tf

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'annotations_json', default=None,
    help=('Path to json file containing the training annotations json for'
          ' the iWildCam2021 competition'))

flags.DEFINE_string(
    'dataset_dir', default=None,
    help=('Path to directory containing training images.'))

flags.mark_flag_as_required('annotations_json')
flags.mark_flag_as_required('dataset_dir')

def _load_dataset_info():
  with tf.io.gfile.GFile(FLAGS.annotations_json, 'r') as json_file:
    json_data = json.load(json_file)
  images = pd.DataFrame(json_data['images'])

  return images

def _read_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.io.decode_jpeg(image, channels=3)
  return image

def main(_):
  images = _load_dataset_info()

  for _, row in images.iterrows():
    file_path = os.path.join(FLAGS.dataset_dir, row.file_name)
    _read_image(file_path)
    print(file_path)

if __name__ == '__main__':
  app.run(main)
