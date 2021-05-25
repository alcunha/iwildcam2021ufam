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

r"""Tool to generate features for each bbox to run deepsort.

Set the environment variable PYTHONHASHSEED to a reproducible value
before you start the python process to ensure that the model trains
or infers with reproducibility
"""
import json
import random
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from reid_extractor import ReID_Inference
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

flags.DEFINE_string(
    'base_model_weights', default='imagenet',
    help=('Path to h5 weights file to be loaded into the base model during'
          ' model build procedure.'))

flags.DEFINE_bool(
    'use_classifier_features', default=True,
    help=('Use features from classifier model'))

flags.DEFINE_list(
    'reid_models', default=None,
    help=('List of omnix reid models to generate features'))

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

flags.DEFINE_float(
    'min_confidence', default=0.9,
    help=('Min confidence for Megadetector predictions.'))

flags.DEFINE_string(
    'features_file', default=None,
    help=('Path to file where data will be saved to.'))

flags.DEFINE_integer(
    'log_frequence', default=500,
    help=('Log prediction every n steps'))

if 'random_seed' not in list(FLAGS):
  flags.DEFINE_integer(
      'random_seed', default=42,
      help=('Random seed for reproductible experiments'))

flags.mark_flag_as_required('test_info_json')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('megadetector_results_json')
flags.mark_flag_as_required('features_file')

BATCH_SIZE = 1

def _build_input_data():
  input_data = dataloader.BBoxFeaturesInputProcessor(
    dataset_json=FLAGS.test_info_json,
    dataset_dir=FLAGS.dataset_dir,
    megadetector_results_json=FLAGS.megadetector_results_json,
    batch_size=BATCH_SIZE,
    output_size=FLAGS.input_size,
    min_confidence=FLAGS.min_confidence,
    seed=FLAGS.random_seed)

  return input_data.make_source_dataset()

def _load_model():
  return model_builder.create(model_name=FLAGS.model_name,
                              input_size=FLAGS.input_size,
                              base_model_weights=FLAGS.base_model_weights)

def _load_reid_models():
  if FLAGS.reid_models is not None:
    models = [ReID_Inference(model_file) for model_file in FLAGS.reid_models]
  else:
    models = []

  return models

def _reid_preprocess(batch):
  mean = np.asarray([123.675,116.280,103.530])
  std = np.asarray([57.0,57.0,57.0])
  images = []
  for bb_img in batch.numpy():
    bb_img = bb_img[:, :, ::-1]
    bb_img = (bb_img-mean)/std
    bb_img = np.transpose(bb_img, (2, 0, 1)).astype(np.float32)
    images.append(bb_img)

  return np.asarray(images)

def generate_features(model, reid_models, dataset):
  features_list = []
  count = 0

  for batch, metadata in dataset:
    features = []
    if FLAGS.use_classifier_features:
      feats = model(batch, training=False)
      features += feats.numpy()[0].tolist()

    reid_batch = _reid_preprocess(batch)
    for reid_model in reid_models:
      feats = reid_model(reid_batch)
      features += feats[0].tolist()

    bbox_info = {
      'img_id': metadata[0].numpy()[0].decode("utf-8"),
      'category': metadata[1].numpy()[0].decode("utf-8"),
      'bbox_tlwh': metadata[2].numpy()[0].tolist(),
      'conf': metadata[3].numpy()[0],
      'features': features
    }
    features_list.append(bbox_info)

    if count % FLAGS.log_frequence == 0:
      tf.compat.v1.logging.info('Finished eval step %d' % count)
    count += 1

  return features_list

def set_random_seeds():
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.random.set_seed(FLAGS.random_seed)

def main(_):
  set_random_seeds()

  if FLAGS.reid_models is None and not FLAGS.use_classifier_features:
    raise RuntimeError('You must use at least one reid model or a classifier'
                       ' to generate feature. Use --reid_models or'
                       ' --use_classifier_features')

  dataset = _build_input_data()
  model = _load_model()
  reid_models = _load_reid_models()

  features = generate_features(model, reid_models, dataset)
  with open(FLAGS.features_file, 'w') as fp:
    json.dump(features, fp)

  print("Features saved to %s" % FLAGS.features_file)

if __name__ == '__main__':
  app.run(main)
