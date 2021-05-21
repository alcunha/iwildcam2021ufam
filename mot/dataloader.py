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

import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE

class BBoxFeaturesInputProcessor:
  def __init__(self,
              dataset_json,
              dataset_dir,
              megadetector_results_json,
              batch_size,
              min_confidence=0.9,
              output_size=224,
              batch_drop_remainder=True,
              seed=None):
    self.dataset_json = dataset_json
    self.dataset_dir = dataset_dir
    self.megadetector_results_json = megadetector_results_json
    self.batch_size = batch_size
    self.min_confidence = min_confidence
    self.output_size = output_size
    self.batch_drop_remainder = batch_drop_remainder
    self.seed = seed

  def _split_bbox(self, img_detections):
    img_id = img_detections['id']
    detections = img_detections['detections']
    bboxes = []

    for det in detections:
      det_parsed = det.copy()
      det_parsed['id'] = img_id

      bbox_x, bbox_y, bbox_width, bbox_height = det_parsed['bbox']
      det_parsed['bbox_x'] = bbox_x
      det_parsed['bbox_y'] = bbox_y
      det_parsed['bbox_width'] = bbox_width
      det_parsed['bbox_height'] = bbox_height
      del det_parsed['bbox']

      bboxes.append(det_parsed)

    return bboxes

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])

    with tf.io.gfile.GFile(self.megadetector_results_json, 'r') as json_file:
      json_data = json.load(json_file)
    megadetector_preds = pd.DataFrame(json_data['images'])

    bboxes = []
    for _, row in megadetector_preds.iterrows():
      bboxes += self._split_bbox(row)

    megadetector_bboxes = pd.DataFrame(bboxes)
    megadetector_bboxes = pd.merge(megadetector_bboxes,
                                   images,
                                   how='left',
                                   on='id')

    return megadetector_bboxes

  def make_source_dataset(self):
    metadata = self._load_metadata()
    metadata = metadata[metadata.conf >= self.min_confidence].copy()

    dataset = tf.data.Dataset.from_tensor_slices((
      metadata.file_name,
      metadata.id,
      metadata.category,
      metadata.conf,
      metadata.bbox_x,
      metadata.bbox_y,
      metadata.bbox_width,
      metadata.bbox_height,
      metadata.height,
      metadata.width))

    def _bbox2tlwh(bbox_x, bbox_y, bbox_width, bbox_height, height, width):
      def castMultiply(x, y):
        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)
        return tf.cast(x * y, dtype=tf.int32)

      bbox_x = castMultiply(bbox_x, width)
      bbox_y = castMultiply(bbox_y, height)
      bbox_width = castMultiply(bbox_width, width)
      bbox_height = castMultiply(bbox_height, height)

      return tf.stack([bbox_x, bbox_y, bbox_width, bbox_height], axis=0)

    def _decode_bboxes(bbox_x, bbox_y, bbox_width, bbox_height):
      xmin = bbox_x
      ymin = bbox_y
      xmax = xmin + bbox_width
      ymax = ymin + bbox_height

      bbox = tf.stack([xmin, ymin, xmax, ymax], axis=0)
      bbox = tf.reshape(bbox, shape=[1, 1, 4])
      bbox = tf.cast(bbox, dtype=tf.float32)

      return bbox

    def _load_and_preprocess_image(filename,
                                   img_id,
                                   category,
                                   conf,
                                   bbox_x,
                                   bbox_y,
                                   bbox_width,
                                   bbox_height,
                                   height,
                                   width):
      bbox_tlwh = _bbox2tlwh(bbox_x, bbox_y, bbox_width, bbox_height,
                             height, width)
      bbox = _decode_bboxes(bbox_x, bbox_y, bbox_width, bbox_height)
      image = tf.io.read_file(self.dataset_dir + filename)
      image = tf.io.decode_jpeg(image, channels=3)
      image = preprocessing.preprocess_image(image,
                                  output_size=self.output_size,
                                  bboxes=bbox,
                                  use_square_crop=True,
                                  is_training=False)

      return image, (img_id, category, bbox_tlwh, conf)

    dataset = dataset.map(_load_and_preprocess_image,
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
