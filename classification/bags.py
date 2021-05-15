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

import numpy as np
import pandas as pd
import tensorflow as tf

class BalancedGroupSoftmax:
  def __init__(self,
               dataset_json,
               category_map,
               empty_class_id,
               selected_locations=None,
               n_groups=4,
               sl_max_groups=[0, 10, 100, 1000, 2**100],
               beta_others=8.0):
    self.n_groups = n_groups
    self.sl_max_groups = sl_max_groups
    self.beta_others = beta_others
    self.category_map = category_map
    self.empty_class_id = empty_class_id
    self.label2binlabel = {}
    self.groups_counts = [1] * (n_groups + 1)
    self.predict_tables = []

    dataset_df = self._load_dataset(dataset_json, selected_locations)
    self._generate_binlabel_idx(dataset_df)
    self._generate_predict_tables()

  def _load_dataset(self, dataset_json, selected_locations):
    with tf.io.gfile.GFile(dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])
    annotations = pd.DataFrame(json_data['annotations'])
    images = pd.merge(images,
                        annotations[["image_id", "category_id"]],
                        how='left',
                        left_on='id',
                        right_on='image_id')

    if selected_locations is not None:
      images = images[images.location.isin(selected_locations)]
      images = images.copy()

    return images

  def _get_group(self, instances_count):
    for group, group_max in enumerate(self.sl_max_groups):
      if instances_count < group_max:
        return group
    return 0

  def _generate_binlabel_idx(self, dataset_df):
    categories = list(range(self.category_map.get_num_classes()))

    #group 0 is only bg/fg
    self.groups_counts[0] = 2
    empty_class = self.category_map.category_to_index(self.empty_class_id)
    self.label2binlabel[empty_class] = [1] + [0] * (self.n_groups)
    categories.remove(empty_class)

    #nonempty categories
    for categ in categories:
      categ_id = self.category_map.index_to_category(categ)
      instances_count = len(dataset_df[dataset_df.category_id == categ_id])
      group_id = self._get_group(instances_count)

      binlabel = [0] * (self.n_groups + 1)
      binlabel[group_id] = self.groups_counts[group_id]
      self.groups_counts[group_id] += 1
      self.label2binlabel[categ] = binlabel

  def _generate_predict_tables(self):
    for i in range(self.n_groups + 1):
      self.predict_tables.append(
        np.zeros(shape=(self.groups_counts[i],
                        self.category_map.get_num_classes())))

    for label, binlabel in self.label2binlabel.items():
      group = np.asarray(binlabel).argmax()
      self.predict_tables[group][binlabel[group]][label] = 1.0

  def create_classif_header(self, head_features):
    outputs = []

    for group_count in self.groups_counts:
      output = tf.keras.layers.Dense(group_count,
                                     activation='softmax')(head_features)
      outputs.append(output)

    return outputs

  def _create_map_layer(self, inputs, n_inputs, n_outputs, weights):
    map_layer = tf.keras.layers.Dense(n_outputs, use_bias=False)
    map_layer(tf.convert_to_tensor(np.ones((1, n_inputs)), dtype=tf.float32))
    map_layer.set_weights([weights])

    return map_layer(inputs)

  def create_prediction_model(self, trained_model):
    fg_prob_map = np.array([np.ones(self.category_map.get_num_classes()),
                            np.zeros(self.category_map.get_num_classes())])
    fg_prob = self._create_map_layer(trained_model.outputs[0],
                                     self.groups_counts[0],
                                     self.category_map.get_num_classes(),
                                     fg_prob_map)

    mapped_predictions = []
    for output, group_size, predict_tbl in zip(trained_model.outputs,
                                               self.groups_counts,
                                               self.predict_tables):
      layer_map = self._create_map_layer(output,
                                          group_size,
                                          self.category_map.get_num_classes(),
                                          predict_tbl)
      mapped_predictions.append(layer_map)

    scaled_mapped_predictions = [mapped_predictions[0]]
    for map_pred in mapped_predictions[1:]:
      scaled_map_pred = tf.keras.layers.Multiply()([map_pred, fg_prob])
      scaled_mapped_predictions.append(scaled_map_pred)

    preds = tf.keras.layers.Add()(scaled_mapped_predictions)
    model = tf.keras.models.Model(inputs=trained_model.inputs, outputs=preds)

    return model

  def process_label(self, label):
    def _get_idx_label(label):
      categ_id = self.category_map.category_to_index(label.numpy())
      binlabels = self.label2binlabel[categ_id]
      binlabels_one_hot = []

      for idx, binlabel in enumerate(binlabels):
        one_hot = np.zeros(self.groups_counts[idx])
        one_hot[binlabel] = 1
        binlabels_one_hot.append(one_hot)

      return binlabels_one_hot
    labels = tf.py_function(func=_get_idx_label,
                            inp=[label],
                            Tout=([tf.float32]*(self.n_groups+1)))
    labels = [tf.ensure_shape(label, shape=(self.groups_counts[i],))
              for i, label in enumerate(labels)]

    return tuple(labels)

  def generate_balancing_mask(self, labels):
    batch_size = tf.shape(labels[0])[0]
    masks = []

    #for the bg/fg group we use all instances
    mask0 = tf.ones(shape=(batch_size,))
    masks.append(mask0)

    def _get_max(labels, batch_size):
      labels = labels.numpy()
      others = labels[:,0]
      fg = 1.0 - others
      fg_num = np.sum(fg)

      if fg_num == 0:
        return np.zeros(batch_size)

      others_num = batch_size - fg_num
      others_sample_num = int(fg_num * self.beta_others)

      if others_sample_num > others_num:
        return np.ones(batch_size)
      else:
        sample_idx = np.random.choice(others.nonzero()[0],
                                      (others_sample_num, ), replace=False)
        fg[sample_idx] = 1.0

      return fg

    for i in range(1, self.n_groups + 1):
      mask = tf.py_function(func=_get_max,
                            inp=[labels[i], batch_size],
                            Tout=tf.float32)
      masks.append(mask)

    return tuple(masks)
