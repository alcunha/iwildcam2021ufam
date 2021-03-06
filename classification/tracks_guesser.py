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

class TrackGuesser:
  def __init__(self,
               dataset_json,
               megadetector_results_json,
               megadetector_threshold=0.9,
               track_nobbox=False,):
    self.dataset_json = dataset_json
    self.megadetector_results_json = megadetector_results_json
    self.megadetector_threshold = megadetector_threshold
    self.track_nobbox = track_nobbox
    self._load_data(dataset_json, megadetector_results_json)
    self._count_detections()

  def _load_data(self, dataset_json, megadetector_results_json):
    with tf.io.gfile.GFile(dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    dataset_info = pd.DataFrame(json_data['images'])

    with tf.io.gfile.GFile(megadetector_results_json, 'r') as json_file:
      json_data = json.load(json_file)
    megadetector_preds = pd.DataFrame(json_data['images'])

    dataset_info = pd.merge(dataset_info,
                            megadetector_preds,
                            how='left',
                            on='id')
    for row in dataset_info.loc[dataset_info.detections.isna()].index:
      data.at[row, 'detections'] = []
    self.dataset_info = dataset_info

  def _count_detections(self):
    def count_dets(row):
      count = 0
      for bbox in row['detections']:
        if bbox['conf'] > self.megadetector_threshold:
          count += 1
      return count
    self.dataset_info['detections_count'] = self.dataset_info.apply(count_dets,
                                                                    axis=1)

  def _max_bbox_per_seq(self, seq_id):
    seq_info = self.dataset_info[self.dataset_info.seq_id == seq_id]
    return seq_info['detections_count'].max()

  def _prepare_extra_tracks(self, extra_tracks_list):
    with tf.io.gfile.GFile(self.dataset_json, 'r') as json_file:
      json_data = json.load(json_file)
    images = pd.DataFrame(json_data['images'])

    tracks = pd.DataFrame(extra_tracks_list)
    tracks = pd.merge(tracks,
                      images,
                      how='left',
                      left_on='img_id',
                      right_on='id')

    return tracks

  def _create_track_nobbox(self, seq_id):
    extra_track = []
    images = self.dataset_info[self.dataset_info.seq_id == seq_id]
    for _, row in images.iterrows():
      bbox_info = {
        'seq_id': seq_id,
        'img_id': row['id'],
        'track_id': seq_id + 'a',
        'bbox_tlwh': [0.0, 0.0, row['width'], row['height']]
      }
      extra_track.append(bbox_info)
    return extra_track

  def _create_single_track_from_bboxes(self, seq_id):
    extra_track = []
    images = self.dataset_info[self.dataset_info.seq_id == seq_id]
    for _, row in images.iterrows():
      dets = row['detections']
      if len(dets) > 0 and dets[0]['conf'] > self.megadetector_threshold:
        bbox = dets[0]['bbox']
        bbox_info = {
          'seq_id': seq_id,
          'img_id': row['id'],
          'track_id': seq_id + 'a',
          'bbox_tlwh': [bbox[0]*row['width'], bbox[1]*row['height'],
                        bbox[2]*row['width'], bbox[3]*row['height']]
        }
        extra_track.append(bbox_info)

    return extra_track

  def guess_tracks(self, confirmed_tracks_df):
    extra_tracks = []
    empty_extra_tracks = 0
    non_confirmed_tracks = 0

    for seq_id in list(self.dataset_info.seq_id.unique()):
      max_bbox = self._max_bbox_per_seq(seq_id)
      num_confirmed_tracks = len(confirmed_tracks_df[
                                        confirmed_tracks_df.seq_id_x==seq_id])

      if max_bbox == 0:
        if self.track_nobbox:
          extra_track = self._create_track_nobbox(seq_id)
          empty_extra_tracks += 1
          extra_tracks += extra_track
      elif max_bbox == 1 and num_confirmed_tracks == 0:
        extra_track = self._create_single_track_from_bboxes(seq_id)
        non_confirmed_tracks += 1
        extra_tracks += extra_track

    print("Confirmed tracks: %d" % len(confirmed_tracks_df.track_id.unique()))
    print("Extra tracks for sequences without bboxes: %d" % empty_extra_tracks)
    print("Extra tracks for sequences with bboxes: %d" % non_confirmed_tracks)

    if len(extra_tracks) > 0:
      extra_tracks_df = self._prepare_extra_tracks(extra_tracks)
      return pd.concat([confirmed_tracks_df, extra_tracks_df],
                       ignore_index=True)
    else:
      return confirmed_tracks_df
