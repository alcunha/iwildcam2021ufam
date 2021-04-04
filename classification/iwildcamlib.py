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

from absl import flags

import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'megadetector_threshold', default=0.9,
    help=('Threshold to use megadetector bounding box on individual counts'))

class CategoryMap:
  def __init__(self, dataset_json):
    with open(dataset_json) as json_file:
      data = json.load(json_file)

    category2idx = {}
    idx2category = {}
    category2name = {}
    category_list = []

    for idx, category in enumerate(data['categories']):
      category2idx[category['id']] = idx
      idx2category[idx] = category['id']
      category2name[category['id']] = category['name']
      category_list.append(category['id'])

    self.category2idx = category2idx
    self.idx2category = idx2category
    self.category2name = category2name
    self.num_classes = len(self.category2idx)
    self.category_list = category_list

  def category_to_index(self, category):
    return self.category2idx[category]

  def index_to_category(self, index):
    return self.idx2category[index]

  def category_to_name(self, category):
    return self.category2name[category]

  def get_category_list(self):
    return self.category_list

  def get_num_classes(self):
    return self.num_classes

def _load_seq_info(instance_ids, predictions, test_info_json):
  with tf.io.gfile.GFile(test_info_json, 'r') as json_file:
    json_data = json.load(json_file)
  test_set = pd.DataFrame(json_data['images'])
  preds = pd.DataFrame(list(zip(instance_ids, predictions)),
                       columns=['id', 'Category'])
  preds = pd.merge(test_set, preds, how='left', on='id')
  preds['Category'] = preds['Category'].fillna(0)
  preds['Category'] = preds['Category'].astype(int)
  preds = preds[['id', 'seq_id', 'Category']].copy()

  return preds

def _load_megadetector_counts(test_info_json, megadetector_results_json):
  with tf.io.gfile.GFile(test_info_json, 'r') as json_file:
    json_data = json.load(json_file)
  test_set = pd.DataFrame(json_data['images'])

  with tf.io.gfile.GFile(megadetector_results_json, 'r') as json_file:
    json_data = json.load(json_file)
  mega_detector_results = pd.DataFrame(json_data['images'])

  data = pd.merge(test_set, mega_detector_results, how='left', on='id')
  for row in data.loc[data.detections.isna()].index:
    data.at[row, 'detections'] = []

  def count_detections(row):
    count = 0
    for bbox in row['detections']:
      if bbox['conf'] > FLAGS.megadetector_threshold:
        count += 1
    return count
  data['detections_count'] = data.apply(count_detections, axis=1)

  seq_counts = {}
  for seq_id in test_set.seq_id.unique():
    max_count = data[data.seq_id == seq_id]['detections_count'].max()
    seq_counts[seq_id] = max_count

  return seq_counts

def _get_majority_vote_prediction(seq_preds):
  preds = list(seq_preds.Category.values)
  return max(set(preds), key=preds.count)

def _get_seq_pred(predictions, seq_id, ignore_empty_images=True):
  seq_preds = predictions[predictions.seq_id == seq_id]

  if ignore_empty_images:
    seq_preds = seq_preds[~(seq_preds.Category == 0)]

  if len(seq_preds) > 0:
    return _get_majority_vote_prediction(seq_preds)

  return 0

def _predict_by_seq(predictios):
  seq_preds = {seq_id: _get_seq_pred(predictios, seq_id)
               for seq_id in predictios.seq_id.unique()}
  return seq_preds

def _generate_zero_submission(seq_ids, categories):
  sub = pd.DataFrame(seq_ids, columns=['Id'])
  for categ in categories[1:]:
    column = 'Predicted' + str(categ)
    sub[column] = 0

  return sub

def _generate_df_submission(seq_preds, seq_counts, category_map):
  categories = category_map.get_category_list()
  submission = _generate_zero_submission(list(seq_preds.keys()), categories)
  for seq_id in seq_preds.keys():
    pred = seq_preds[seq_id]
    if pred > 0:
      column = 'Predicted' + str(pred)
      submission.loc[submission.Id == seq_id, column] = max(1,
                                                            seq_counts[seq_id])

  return submission

def generate_submission(instance_ids, predictions, category_map,
                        test_info_json, megadetector_results_json, csv_file):
  predictions = _load_seq_info(instance_ids, predictions, test_info_json)
  seq_counts = _load_megadetector_counts(test_info_json,
                                         megadetector_results_json)
  seq_preds = _predict_by_seq(predictions)
  df = _generate_df_submission(seq_preds, seq_counts, category_map)
  df.to_csv(csv_file, index=False, header=True, sep=',')
