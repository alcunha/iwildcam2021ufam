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

r"""Tool to train classifiers.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import os
import random

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf

from iwildcamlib import CategoryMap
import dataloader
import model_builder
import utils

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during training.'))

flags.DEFINE_bool(
    'fix_resolution', default=False,
    help=('Apply the fix train-test resolution: fine-tune only the last layer'
          ' and uses test data augmentation. Use the --input_size option'
          ' to specify the test input resolution.'))

flags.DEFINE_string(
    'load_checkpoint', default=None,
    help=('Path to weights checkpoint to be loaded into the model'))

flags.DEFINE_string(
    'annotations_json', default=None,
    help=('Path to json file containing the training annotations json for'
          ' the iWildCam2021 competition'))

flags.DEFINE_string(
    'dataset_dir', default=None,
    help=('Path to directory containing training images.'))

flags.DEFINE_string(
    'megadetector_results_json', default=None,
    help=('Path to json file containing megadetector results.'))

flags.DEFINE_integer(
    'randaug_num_layers', default=None,
    help=('Number of operations to be applied by Randaugment'))

flags.DEFINE_integer(
    'randaug_magnitude', default=None,
    help=('Magnitude for operations on Randaugment.'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments')
  )

flags.mark_flag_as_required('annotations_json')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('megadetector_results_json')

def build_input_data(category_map, is_training):
  input_data =  dataloader.JsonWBBoxInputProcessor(
    dataset_json=FLAGS.annotations_json,
    dataset_dir=FLAGS.dataset_dir,
    megadetector_results_json=FLAGS.megadetector_results_json,
    batch_size=FLAGS.batch_size,
    category_map=category_map,
    is_training=is_training,
    use_eval_preprocess=FLAGS.fix_resolution,
    output_size=FLAGS.input_size,
    randaug_num_layers=FLAGS.randaug_num_layers,
    randaug_magnitude=FLAGS.randaug_magnitude,
    seed=FLAGS.random_seed,
  )

  return input_data.make_source_dataset()

def get_model(num_classes):
  model = model_builder.create(
    model_name=FLAGS.model_name,
    num_classes=num_classes,
    input_size=FLAGS.input_size,
    freeze_layers=FLAGS.fix_resolution,
    seed=FLAGS.random_seed)

  return model

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  if utils.xor(FLAGS.randaug_num_layers is None,
              FLAGS.randaug_magnitude is None):
    raise RuntimeError('To apply Randaugment during training you must specify'
                      ' both --randaug_num_layers and --randaug_magnitude')

  set_random_seeds()

  category_map = CategoryMap(FLAGS.annotations_json)
  train_dataset = build_input_data(category_map, is_training=True)

  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  with strategy.scope():
    model = get_model(category_map.get_num_classes())

  model.summary()

  if FLAGS.load_checkpoint is not None:
    checkpoint_path = os.path.join(FLAGS.load_checkpoint, "ckp")
    model.load_weights(checkpoint_path)

if __name__ == '__main__':
  app.run(main)
