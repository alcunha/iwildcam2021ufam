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
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class JsonWBBoxInputProcessor:
  def __init__(self,
              dataset_json,
              dataset_dir,
              megadetector_results_json,
              batch_size,
              num_classes,
              default_empty_label=0,
              is_training=False,
              use_eval_preprocess=False,
              output_size=224,
              resize_with_pad=False,
              randaug_num_layers=None,
              randaug_magnitude=None,
              use_fake_data=False,
              provide_instance_id=False,
              seed=None):
    self.dataset_json = dataset_json
    self.dataset_dir = dataset_dir
    self.megadetector_results_json = megadetector_results_json
    self.batch_size = batch_size
    self.is_training = is_training
    self.output_size = output_size
    self.resize_with_pad = resize_with_pad
    self.num_classes = num_classes
    self.default_empty_label = default_empty_label
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude
    self.use_fake_data = use_fake_data
    self.provide_instance_id = provide_instance_id
    self.preprocess_for_train = is_training and not use_eval_preprocess
    self.seed = seed
    self.num_instances = 0

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])
    if 'annotations' in json_data.keys():
      annotations = pd.DataFrame(json_data['annotations'])
      images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')
    else:
      images['category_id'] = self.default_empty_label

    with tf.io.gfile.GFile(self.megadetector_results_json, 'r') as json_file:
      json_data = json.load(json_file)
    megadetector_preds = pd.DataFrame(json_data['images'])
    images = pd.merge(images,
                      megadetector_preds,
                      how='left',
                      on='id')

    return images

  def _prepare_bboxes(self, metadata):
    def _get_first_bbox(row):
      bbox = row['detections']
      bbox = bbox[0]['bbox'] if len(bbox) > 0 else [0.0, 0.0, 1.0, 1.0]
      return bbox

    metadata['detections'] = metadata['detections'].apply(
                            lambda d: d if isinstance(d, list) else [])
    metadata['bbox'] = metadata.apply(_get_first_bbox, axis=1)
    bboxes = pd.DataFrame(metadata.bbox.tolist(),
                    columns=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])

    return bboxes.to_dict('list')

  def make_source_dataset(self):
    metadata = self._load_metadata()
    bboxes = self._prepare_bboxes(metadata)

    self.num_instances = len(metadata.file_name)

    dataset = tf.data.Dataset.from_tensor_slices((
      metadata.file_name,
      bboxes,
      metadata.category_id
    ))

    if self.is_training:
      dataset = dataset.shuffle(self.num_instances, seed=self.seed)
      dataset = dataset.repeat()

    def _load_and_preprocess_image(filename, bboxes, label):
      image = tf.io.read_file(self.dataset_dir + filename)
      image = tf.io.decode_jpeg(image, channels=3)

      return image, bboxes, label

    dataset = dataset.map(_load_and_preprocess_image,
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)

    if self.use_fake_data:
      dataset.take(1).repeat()

    return dataset
