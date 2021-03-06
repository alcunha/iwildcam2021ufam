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
import math

from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf

import preprocessing
import utils

flags.DEFINE_string(
    'loc_encode', default='encode_cos_sin',
    help=('Encoding type for location coordinates'))

flags.DEFINE_string(
    'date_encode', default='encode_cos_sin',
    help=('Encoding type for date'))

flags.DEFINE_bool(
    'use_date_feats', default=True,
    help=('Include date features to the encoded coordinates inputs'))

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS

def _encode_feat(feat, encode):
  if encode == 'encode_cos_sin':
    return tf.sin(math.pi*feat), tf.cos(math.pi*feat)
  else:
    raise RuntimeError('%s not implemented' % encode)

  return feat

class JsonWBBoxInputProcessor:
  def __init__(self,
              dataset_json,
              dataset_dir,
              megadetector_results_json,
              batch_size,
              category_map,
              bal_group_softmax=None,
              crop_mode='bbox',
              conf_threshold=0.6,
              selected_locations=None,
              default_empty_label=0,
              is_training=False,
              use_eval_preprocess=False,
              output_size=224,
              resize_with_pad=False,
              randaug_num_layers=None,
              randaug_magnitude=None,
              use_fake_data=False,
              provide_validity_info_output=False,
              provide_coord_date_encoded_input=False,
              provide_instance_id=False,
              batch_drop_remainder=True,
              seed=None):
    self.dataset_json = dataset_json
    self.dataset_dir = dataset_dir
    self.megadetector_results_json = megadetector_results_json
    self.batch_size = batch_size
    self.category_map = category_map
    self.bal_group_softmax = bal_group_softmax
    self.crop_mode = crop_mode
    self.conf_threshold = conf_threshold
    self.selected_locations = selected_locations
    self.is_training = is_training
    self.output_size = output_size
    self.resize_with_pad = resize_with_pad
    self.default_empty_label = default_empty_label
    self.randaug_num_layers = randaug_num_layers
    self.randaug_magnitude = randaug_magnitude
    self.use_fake_data = use_fake_data
    self.provide_validity_info_output = provide_validity_info_output
    self.provide_coord_date_encoded_input = provide_coord_date_encoded_input
    self.provide_instance_id = provide_instance_id
    self.preprocess_for_train = is_training and not use_eval_preprocess
    self.batch_drop_remainder = batch_drop_remainder
    self.seed = seed
    self.num_instances = 0

  def _validate_location_info_from_metadata(self, metadata_df):
    if self.provide_coord_date_encoded_input:
      if 'longitude' not in metadata_df.columns:
        raise RuntimeError('Logintude info does not exists on dataset_json.'
                          ' Please add to json.')
      if 'latitude' not in metadata_df.columns:
        raise RuntimeError('Latitude info does not exists on dataset_json.'
                          ' Please add to json')
      if 'date' not in metadata_df.columns:
        raise RuntimeError('Date info does not exists on dataset_json.'
                          ' Please add to json.')

      metadata_df['date_c'] = metadata_df.apply(
                            lambda row: utils.date2float(row['date']), axis=1)
    else:
      metadata_df['longitude'] = 0
      metadata_df['latitude'] = 0
      metadata_df['date_c'] = 0

    metadata_df['valid'] = ~metadata_df.longitude.isna()

    return metadata_df

  def _explode_bboxes(self, metadata_df):
    detections = metadata_df['detections'].apply(pd.Series).reset_index()
    detections = detections.melt(id_vars='index').dropna()[['index', 'value']]
    detections = detections.set_index('index')
    detections = detections[detections.apply(
                      lambda row: row['value']['conf'] >= self.conf_threshold,
                      axis=1)].copy()

    df_expanded = pd.merge(detections,
                    metadata_df.loc[:, metadata_df.columns != 'detections'],
                    left_index=True,
                    right_index=True,
                    how='outer')
    df_expanded = df_expanded.rename(columns={'value': 'detections'})
    df_expanded = df_expanded.reset_index()
    for row in df_expanded.loc[~df_expanded.detections.isna()].index:
      df_expanded.at[row, 'detections'] = [df_expanded.at[row, 'detections']]
    for row in df_expanded.loc[df_expanded.detections.isna()].index:
      df_expanded.at[row, 'detections'] = []

    return df_expanded.copy()

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

    images = self._validate_location_info_from_metadata(images)

    return images

  def _prepare_bboxes(self, metadata):
    def _get_first_bbox(row):
      bbox = row['detections']
      if len(bbox) > 0 and bbox[0]['conf'] > self.conf_threshold:
        bbox = bbox[0]['bbox']
      else:
        bbox = [0.0, 0.0, 1.0, 1.0]
      return bbox

    metadata['detections'] = metadata['detections'].apply(
                            lambda d: d if isinstance(d, list) else [])
    metadata['bbox'] = metadata.apply(_get_first_bbox, axis=1)
    bboxes = pd.DataFrame(metadata.bbox.tolist(),
                    columns=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])

    return bboxes.to_dict('list')

  def make_source_dataset(self):
    self.num_classes = self.category_map.get_num_classes()
    metadata = self._load_metadata()
    if self.selected_locations is not None:
      metadata = metadata[metadata.location.isin(self.selected_locations)]
      metadata = metadata.copy()

    if self.is_training and self.crop_mode == 'bbox':
      metadata = self._explode_bboxes(metadata)

    bboxes = self._prepare_bboxes(metadata)

    self.num_instances = len(metadata.file_name)

    dataset = tf.data.Dataset.from_tensor_slices((
      metadata.file_name,
      bboxes,
      metadata.valid,
      metadata.longitude,
      metadata.latitude,
      metadata.date_c,
      metadata.category_id
    ))

    if self.is_training:
      dataset = dataset.shuffle(self.num_instances, seed=self.seed)
      dataset = dataset.repeat()

    def _decode_bboxes(bboxes):
      xmin = bboxes['bbox_x']
      ymin = bboxes['bbox_y']
      xmax = xmin + bboxes['bbox_width']
      ymax = ymin + bboxes['bbox_height']

      bbox = tf.stack([xmin, ymin, xmax, ymax], axis=0)
      bbox = tf.reshape(bbox, shape=[1, 1, 4])

      return bbox

    def _encode_lat_lon(lat, lon, date):
      lat = _encode_feat(lat, FLAGS.loc_encode)
      lon = _encode_feat(lon, FLAGS.loc_encode)
      if FLAGS.use_date_feats:
        date = date*2.0 - 1.0
        date = _encode_feat(date, FLAGS.date_encode)
        coord_date_encoded = tf.concat([lon, lat, date], axis=0)
      else:
        coord_date_encoded = tf.concat([lon, lat], axis=0)
      coord_date_encoded = tf.cast(coord_date_encoded, tf.float32)

      return coord_date_encoded

    def _load_and_preprocess_image(filename,
                                   bboxes,
                                   valid,
                                   lat,
                                   lon,
                                   date,
                                   label):
      bbox = _decode_bboxes(bboxes)
      image = tf.io.read_file(self.dataset_dir + filename)
      image = tf.io.decode_jpeg(image, channels=3)

      coord_date = _encode_lat_lon(lat, lon, date) \
        if self.provide_coord_date_encoded_input else None

      if self.crop_mode == 'bbox':
        image = preprocessing.preprocess_image(image,
                                    output_size=self.output_size,
                                    bboxes=bbox,
                                    use_square_crop=True,
                                    is_training=self.preprocess_for_train,
                                    resize_with_pad=self.resize_with_pad,
                                    randaug_num_layers=self.randaug_num_layers,
                                    randaug_magnitude=self.randaug_magnitude)
        inputs = (image, coord_date) if self.provide_coord_date_encoded_input \
                                     else image
      elif self.crop_mode == 'full':
        image = preprocessing.preprocess_image(image,
                                    output_size=self.output_size,
                                    bboxes=None,
                                    use_square_crop=False,
                                    is_training=self.preprocess_for_train,
                                    resize_with_pad=self.resize_with_pad,
                                    randaug_num_layers=self.randaug_num_layers,
                                    randaug_magnitude=self.randaug_magnitude)
        inputs = (image, coord_date) if self.provide_coord_date_encoded_input \
                                     else image
      elif self.crop_mode == 'both' or self.crop_mode == 'both_flip':
        image1 = preprocessing.preprocess_image(image,
                                    output_size=self.output_size,
                                    bboxes=bbox,
                                    use_square_crop=True,
                                    is_training=self.preprocess_for_train,
                                    resize_with_pad=self.resize_with_pad,
                                    randaug_num_layers=self.randaug_num_layers,
                                    randaug_magnitude=self.randaug_magnitude)
        image2 = preprocessing.preprocess_image(image,
                                    output_size=self.output_size,
                                    bboxes=None,
                                    use_square_crop=False,
                                    is_training=self.preprocess_for_train,
                                    resize_with_pad=self.resize_with_pad,
                                    randaug_num_layers=self.randaug_num_layers,
                                    randaug_magnitude=self.randaug_magnitude)
        if self.crop_mode == 'both_flip':
          image1_flip = tf.image.flip_left_right(image1)
          image2_flip = tf.image.flip_left_right(image2)
          if self.provide_coord_date_encoded_input:
            inputs = (image1, image1_flip, image2, image2_flip, coord_date)
          else:
            inputs = (image1, image1_flip, image2, image2_flip)
        else:
          if self.provide_coord_date_encoded_input:
            inputs = (image1, image2, coord_date)
          else:
            inputs = (image1, image2)
      else:
        raise ValueError('Invalid crop_mode, used %s.' % self.crop_mode)

      if self.bal_group_softmax is not None:
        label = self.bal_group_softmax.process_label(label)
      else:
        def _get_idx_label(label):
          return self.category_map.category_to_index(label.numpy())
        label = tf.py_function(func=_get_idx_label, inp=[label], Tout=tf.int32)
        label = tf.reshape(label, shape=())
        label = tf.one_hot(label, self.num_classes)

      if self.provide_validity_info_output:
        valid = tf.cast(valid, tf.float32)
        outputs = (label, valid, filename) if self.provide_instance_id \
                                           else (label, valid)
      else:
        outputs = (label, filename) if self.provide_instance_id else label

      return inputs, outputs

    dataset = dataset.map(_load_and_preprocess_image,
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)

    if self.bal_group_softmax is not None:
      def _generate_masks(inputs, outputs):
        masks = self.bal_group_softmax.generate_balancing_mask(outputs)
        return (inputs, outputs, masks)
      dataset = dataset.map(_generate_masks,
                            num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    if self.use_fake_data:
      dataset.take(1).repeat()

    return dataset, self.num_instances

class BBoxInputProcessor:
  def __init__(self,
              dataset_json,
              dataset_dir,
              megadetector_results_json,
              batch_size,
              conf_threshold=0.9,
              output_size=224,
              resize_with_pad=False,
              provide_instance_id=False,
              batch_drop_remainder=True,
              seed=None):
    self.dataset_json = dataset_json
    self.dataset_dir = dataset_dir
    self.megadetector_results_json = megadetector_results_json
    self.batch_size = batch_size
    self.conf_threshold = conf_threshold
    self.output_size = output_size
    self.resize_with_pad = resize_with_pad
    self.provide_instance_id = provide_instance_id
    self.batch_drop_remainder = batch_drop_remainder
    self.seed = seed
    self.num_instances = 0

  def _explode_bboxes(self, metadata_df):
    detections = metadata_df['detections'].apply(pd.Series).reset_index()
    detections = detections.melt(id_vars='index').dropna()[['index', 'value']]
    detections = detections.set_index('index')
    detections = detections[detections.apply(
                      lambda row: row['value']['conf'] >= self.conf_threshold,
                      axis=1)].copy()

    df_expanded = pd.merge(detections,
                    metadata_df.loc[:, metadata_df.columns != 'detections'],
                    left_index=True,
                    right_index=True,
                    how='left')
    df_expanded = df_expanded.rename(columns={'value': 'detections'})
    df_expanded = df_expanded.reset_index()
    for row in df_expanded.loc[~df_expanded.detections.isna()].index:
      df_expanded.at[row, 'detections'] = [df_expanded.at[row, 'detections']]

    return df_expanded.copy()

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])

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
      return row['detections'][0]['bbox']

    metadata['bbox'] = metadata.apply(_get_first_bbox, axis=1)
    bboxes = pd.DataFrame(metadata.bbox.tolist(),
                    columns=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])

    return bboxes.to_dict('list')

  def make_source_dataset(self):
    metadata = self._load_metadata()
    metadata = self._explode_bboxes(metadata)
    bboxes = self._prepare_bboxes(metadata)

    self.num_instances = len(metadata.file_name)

    dataset = tf.data.Dataset.from_tensor_slices((
      metadata.file_name,
      metadata.id,
      bboxes,
    ))

    def _decode_bboxes(bboxes):
      xmin = bboxes['bbox_x']
      ymin = bboxes['bbox_y']
      xmax = xmin + bboxes['bbox_width']
      ymax = ymin + bboxes['bbox_height']

      bbox = tf.stack([xmin, ymin, xmax, ymax], axis=0)
      bbox = tf.reshape(bbox, shape=[1, 1, 4])

      return bbox

    def _load_and_preprocess_image(filename, instance_id, bboxes):
      bbox = _decode_bboxes(bboxes)
      image = tf.io.read_file(self.dataset_dir + filename)
      image = tf.io.decode_jpeg(image, channels=3)

      image = preprocessing.preprocess_image(image,
                                  output_size=self.output_size,
                                  bboxes=bbox,
                                  use_square_crop=True,
                                  is_training=False,
                                  resize_with_pad=self.resize_with_pad)

      if self.provide_instance_id:
        return image, instance_id

      return image

    dataset = dataset.map(_load_and_preprocess_image,
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

class TrackInputProcessor:
  def __init__(self,
               dataset_json,
               dataset_dir,
               tracks_json,
               batch_size,
               tracks_guesser=None,
               num_images=8,
               output_size=224,
               resize_with_pad=False,
               batch_drop_remainder=True):
    self.dataset_json = dataset_json
    self.dataset_dir = dataset_dir
    self.tracks_json = tracks_json
    self.tracks_guesser = tracks_guesser
    self.batch_size = batch_size
    self.num_images = num_images
    self.output_size = output_size
    self.resize_with_pad = resize_with_pad
    self.batch_drop_remainder = batch_drop_remainder

  def _load_metadata(self):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])

    with tf.io.gfile.GFile(self.tracks_json, 'r') as json_file:
      json_data = json.load(json_file)
    tracks = pd.DataFrame(json_data)
    tracks = pd.merge(tracks,
                      images,
                      how='left',
                      left_on='img_id',
                      right_on='id')

    return tracks

  def _process_tracks(self, track_data):
    seqs_list = []
    tracks_list = []
    filenames_list = []
    bboxes_list = []
    tta_list = []
    height_list = []
    width_list = []
    for track_id in list(track_data.track_id.unique()):
      images = track_data[track_data.track_id == track_id]
      if len(images) > self.num_images:
        images = images.sample(self.num_images)

      filenames = []
      bboxes = []
      tta = []
      height = []
      width = []
      # original bboxes
      for _, img in images.iterrows():
        filenames.append(img['file_name'])
        bboxes.append(np.array(img['bbox_tlwh']))
        tta.append(False)
        height.append(img['height'])
        width.append(img['width'])

      #extra bboxes to be used with test time augmentation
      if len(images) < self.num_images:
        images_sample = images.sample(self.num_images - len(images),
                                      replace=True)
        for _, img in images_sample.iterrows():
          filenames.append(img['file_name'])
          bboxes.append(np.array(img['bbox_tlwh']))
          tta.append(True)
          height.append(img['height'])
          width.append(img['width'])

      seqs_list.append(images['seq_id_x'].iloc[0])
      tracks_list.append(track_id)
      filenames_list.append(filenames)
      bboxes_list.append(bboxes)
      tta_list.append(tta)
      height_list.append(height)
      width_list.append(width)

    return (seqs_list, tracks_list, filenames_list, bboxes_list, tta_list, \
             height_list, width_list)

  def make_source_dataset(self):
    metadata = self._load_metadata()
    if self.tracks_guesser is not None:
      metadata = self.tracks_guesser.guess_tracks(metadata)

    dataset = tf.data.Dataset.from_tensor_slices(
        self._process_tracks(metadata))

    def _decode_bbox(bbox, height, width):
      bbox = tf.cast(bbox, tf.float32) / tf.cast([width, height, width, height],
                                                 tf.float32)
      bbox = tf.stack([bbox[:2], bbox[:2] + bbox[2:]], axis=0)
      bbox = tf.reshape(bbox, shape=[1, 1, 4])
      bbox = tf.cast(bbox, dtype=tf.float32)

      return bbox

    def _load_and_preprocess_image(filename,
                                   bbox,
                                   tta,
                                   height,
                                   width):
      image = tf.io.read_file(self.dataset_dir + filename)
      image = tf.io.decode_jpeg(image, channels=3)
      bbox = _decode_bbox(bbox, height, width)
      image = tf.cond(tta,
                      lambda: preprocessing.preprocess_image(
                                    image,
                                    output_size=self.output_size,
                                    bboxes=bbox,
                                    use_square_crop=True,
                                    is_training=True,
                                    resize_with_pad=self.resize_with_pad),
                      lambda: preprocessing.preprocess_image(
                                    image,
                                    output_size=self.output_size,
                                    bboxes=bbox,
                                    use_square_crop=True,
                                    is_training=False,
                                    resize_with_pad=self.resize_with_pad))
      return image

    def _preprocess_track(seq_id,
                          track_id,
                          filenames,
                          bboxes,
                          tta,
                          heights,
                          widths):

      images = [_load_and_preprocess_image(filenames[i],
                                           bboxes[i],
                                           tta[i],
                                           heights[i],
                                           widths[i])
                for i in range(self.num_images)]

      return tuple(images), (seq_id, track_id)
    dataset = dataset.map(_preprocess_track, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=self.batch_drop_remainder)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

class RandSpatioTemporalGenerator:
  def __init__(self, rand_type='spherical'):
    self.rand_type = rand_type

  def _encode_feat(self, feat, encode):
    if encode == 'encode_cos_sin':
      feats = tf.concat([
        tf.sin(math.pi*feat),
        tf.cos(math.pi*feat)], axis=1)
    else:
      raise RuntimeError('%s not implemented' % encode)

    return feats

  def get_rand_samples(self, batch_size):
    if self.rand_type == 'spherical':
      rand_feats = tf.random.uniform(shape=(batch_size, 3),
                                    dtype=tf.float32)
      theta1 = 2.0*math.pi*rand_feats[:,0]
      theta2 = tf.acos(2.0*rand_feats[:,1] - 1.0)
      lat = 1.0 - 2.0*theta2/math.pi
      lon = (theta1/math.pi) - 1.0
      time = rand_feats[:,2]*2.0 - 1.0

      lon = tf.expand_dims(lon, axis=-1)
      lat = tf.expand_dims(lat, axis=-1)
      time = tf.expand_dims(time, axis=-1)
    else:
      raise RuntimeError('%s rand type not implemented' % self.rand_type)

    lon = self._encode_feat(lon, FLAGS.loc_encode)
    lat = self._encode_feat(lat, FLAGS.loc_encode)
    time = self._encode_feat(time, FLAGS.date_encode)

    if FLAGS.use_date_feats:
      return tf.concat([lon, lat, time], axis=1)
    else:
      return tf.concat([lon, lat], axis=1)
