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

import json
import os
import random

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from iwildcamlib import CategoryMap, generate_submission_by_tracks
import bags
import dataloader
import model_builder

os.environ['TF_DETERMINISTIC_OPS'] = '1'

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model_name', default='efficientnet-b0',
    help=('Model name of the archtecture'))

flags.DEFINE_integer(
    'input_size', default=224,
    help=('Input size of the model'))

flags.DEFINE_bool(
    'use_bags', default=False,
    help=('Use Balanced Group Softmax to train model'))

flags.DEFINE_integer(
    'empty_class_id', default=0,
    help=('Empty class id for balanced group softmax'))

flags.DEFINE_integer(
    'num_images_by_track', default=8,
    help=('Number of images used to predict class for each track'))

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size used during training.'))

flags.DEFINE_string(
    'ckpt_dir', default=None,
    help=('Location of the model checkpoint files'))

flags.DEFINE_string(
    'annotations_json', default=None,
    help=('Path to json file containing the training annotations json for'
          ' the iWildCam2021 competition'))

flags.DEFINE_string(
    'train_dataset_split', default=None,
    help=('Path to json file containing the train/validation split based on'
          ' locations.'))

flags.DEFINE_string(
    'test_info_json', default=None,
    help=('Path to json file containing the test information json for'
          ' the iWildCam2021 competition'))

flags.DEFINE_string(
    'dataset_dir', default=None,
    help=('Path to directory containing test images.'))

flags.DEFINE_string(
    'megadetector_results_json', default=None,
    help=('Path to json file containing megadetector results.'))

flags.DEFINE_string(
    'tracks_file', default=None,
    help=('Path to file contained confirmed tracks info.'))

flags.DEFINE_string(
    'submission_file_path', default=None,
    help=('File name to save predictions on iWildCam2020 results format.'))

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('ckpt_dir')
flags.mark_flag_as_required('annotations_json')
flags.mark_flag_as_required('test_info_json')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('megadetector_results_json')
flags.mark_flag_as_required('tracks_file')
flags.mark_flag_as_required('submission_file_path')

def load_train_validation_split():
  if FLAGS.train_dataset_split is None:
    return None, None

  with tf.io.gfile.GFile(FLAGS.train_dataset_split, 'r') as json_file:
    json_data = json.load(json_file)

  return json_data['train'], json_data['validation']

def _load_model(num_classes, bal_group_softmax=None):
  model = model_builder.create(model_name=FLAGS.model_name,
                              num_classes=num_classes,
                              input_size=FLAGS.input_size,
                              unfreeze_layers=0,
                              bags=bal_group_softmax)
  checkpoint_path = os.path.join(FLAGS.ckpt_dir, "ckp")
  model.load_weights(checkpoint_path)

  if bal_group_softmax is not None:
    model = bal_group_softmax.create_prediction_model(model)

  inputs = [tf.keras.Input(shape=(FLAGS.input_size, FLAGS.input_size, 3))
            for x in range(FLAGS.num_images_by_track)]
  outputs = [model(img_input, training=False) for img_input in inputs]
  outputs = tf.keras.layers.Average()(outputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=[outputs])

  return model

def _build_input_data():
  input_data = dataloader.TrackInputProcessor(
    dataset_json=FLAGS.test_info_json,
    dataset_dir=FLAGS.dataset_dir,
    tracks_json=FLAGS.tracks_file,
    batch_size=FLAGS.batch_size,
    num_images=FLAGS.num_images_by_track,
    batch_drop_remainder=False)

  return input_data.make_source_dataset()

def predict_classifier(model, dataset):
  seq_ids = []
  track_ids = []
  predictions = []
  count = 0

  for batch, metadata in dataset:
    pred = model(batch, training=False)
    seqid, trackid = metadata
    seq_ids += list(seqid.numpy())
    track_ids += list(trackid.numpy())
    predictions += list(pred.numpy())

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return seq_ids, track_ids, predictions

def format_instance_id(instance_ids):
  instance_ids = [instanceid.decode("utf-8") for instanceid in instance_ids]

  return instance_ids

def decode_predictions(predictions, category_map):
  preds = [category_map.index_to_category(pred.argmax())
           for pred in predictions]

  return preds

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  category_map = CategoryMap(FLAGS.annotations_json)
  train_loc, _ = load_train_validation_split()
  bal_group_softmax = bags.BalancedGroupSoftmax(
        FLAGS.annotations_json,
        category_map,
        FLAGS.empty_class_id,
        selected_locations=train_loc) if FLAGS.use_bags else None

  dataset = _build_input_data()
  model = _load_model(category_map.get_num_classes(), bal_group_softmax)

  seq_ids, track_ids, predictions = predict_classifier(model, dataset)

  seq_ids = format_instance_id(seq_ids)
  track_ids = format_instance_id(track_ids)
  predictions = decode_predictions(predictions, category_map)

  generate_submission_by_tracks(seq_ids, track_ids, predictions, category_map,
                      FLAGS.test_info_json, FLAGS.submission_file_path)

if __name__ == '__main__':
  app.run(main)
