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

from absl import flags

import tensorflow as tf

import randaugment
import simpleaugment

flags.DEFINE_enum(
    'input_scale_mode', default='float32',
    enum_values=['tf_mode', 'torch_mode', 'uint8', 'float32'],
    help=('Mode for scaling input: tf_mode scales image between -1 and 1;'
          ' torch_mode normalizes inputs using ImageNet mean and std using'
          ' float32 input format; uint8 uses image on scale 0-255; float32'
          ' uses image on scale 0-1'))

flags.DEFINE_bool(
    'use_simple_augment', default=False,
    help=('Use simple data augmentation with random hue, saturation, '
          ' brightness, and rotation. If Randaugment parameters is used, this'
          ' options is ignored.'))

FLAGS = flags.FLAGS

@tf.function
def square_crop(image, bbox):
  img_height, img_width, _ = tf.split(tf.shape(image), num_or_size_splits=3)
  xmin, ymin, xmax, ymax = tf.split(bbox[0][0], num_or_size_splits=4)
  bbox_width = xmax - xmin
  bbox_height = ymax - ymin

  def castMultiply(x, y):
    y = tf.cast(y, dtype=tf.float32)
    return tf.cast(x * y, dtype=tf.int32)

  offset_width = castMultiply(xmin, img_width)
  offset_height = castMultiply(ymin, img_height)
  target_height = castMultiply(bbox_height, img_height)
  target_width = castMultiply(bbox_width, img_width)

  crop_size = tf.maximum(target_height, target_width)
  crop_size = tf.minimum(tf.minimum(crop_size, img_height), img_width)

  center_x = offset_width + target_width//2
  center_y = offset_height + target_height//2

  offset_width_new = tf.maximum(0, center_x - crop_size//2)
  offset_height_new = tf.maximum(0, center_y - crop_size//2)

  offset_width_new = tf.cond(offset_width_new + crop_size > img_width,
                             lambda: img_width - crop_size,
                             lambda: offset_width_new)
  offset_height_new = tf.cond(offset_height_new + crop_size > img_height,
                              lambda: img_height - crop_size,
                              lambda: offset_height_new)

  image = tf.image.crop_to_bounding_box(image,
                                        offset_height_new[0],
                                        offset_width_new[0],
                                        crop_size[0],
                                        crop_size[0])

  return image

def random_crop(image,
                bboxes,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.65, 1],
                min_object_covered=0.5,
                max_attempts=100,
                seed=None):

  if bboxes is None:
    bboxes = tf.constant([0.0, 0.0, 1.0, 1.0],
                        dtype=tf.float32, shape=[1, 1, 4])

  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=bboxes,
      min_object_covered=min_object_covered,
      area_range=area_range,
      aspect_ratio_range=aspect_ratio_range,
      use_image_if_no_bounding_boxes=True,
      max_attempts=max_attempts,
      seed=seed
  )

  offset_height, offset_width, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.crop_to_bounding_box(
    image,
    offset_height,
    offset_width,
    target_height,
    target_width
  )

  return image

def flip(image, seed=None):
  return tf.image.random_flip_left_right(image, seed)

def normalize_image(image):
  tf.compat.v1.logging.info('Normalizing inputs.')
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  mean = tf.constant([0.485, 0.456, 0.406])
  mean = tf.expand_dims(mean, axis=0)
  mean = tf.expand_dims(mean, axis=0)
  image = image - mean

  std = tf.constant([0.229, 0.224, 0.225])
  std = tf.expand_dims(std, axis=0)
  std = tf.expand_dims(std, axis=0)
  image = image/std

  return image

def scale_input_tf_mode(image):
  image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
  image = tf.cast(image, tf.float32)
  image /= 127.5
  image -= 1.

  return image

def scale_input(image):
  if FLAGS.input_scale_mode == 'torch_mode':
    return normalize_image(image)
  elif FLAGS.input_scale_mode == 'tf_mode':
    return scale_input_tf_mode(image)
  elif FLAGS.input_scale_mode == 'uint8':
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)
  else:
    return tf.image.convert_image_dtype(image, dtype=tf.float32)

def resize_image(image, output_size, resize_with_pad=False):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if resize_with_pad:
    image = tf.image.resize_with_pad(image, output_size, output_size)
  else:
    image = tf.image.resize(image, size=(output_size, output_size))

  return image

def preprocess_for_train(image,
                        output_size,
                        resize_with_pad=False,
                        bboxes=None,
                        use_square_crop=False,
                        randaug_num_layers=None,
                        randaug_magnitude=None,
                        seed=None):

  if output_size is None:
    raise RuntimeError('Output size cannot be None for image preprocessing'
                       ' during training.')

  if use_square_crop:
    if bboxes is not None:
      image = square_crop(image, bboxes)
  else:
    image = random_crop(image, bboxes=None, seed=seed)
  image = resize_image(image, output_size, resize_with_pad)
  image = flip(image, seed)

  if randaug_num_layers is not None and randaug_magnitude is not None:
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = randaugment.distort_image_with_randaugment(image,
                                                       randaug_num_layers,
                                                       randaug_magnitude)
  else:
    if FLAGS.use_simple_augment:
      image = simpleaugment.distort_image_with_simpleaugment(image, seed)

  image = scale_input(image)

  return image

def preprocess_for_eval(image,
                        output_size,
                        bbox=None,
                        use_square_crop=False,
                        resize_with_pad=False):
  if use_square_crop and bbox is not None:
    image = square_crop(image, bbox)

  if output_size is not None:
    image = resize_image(image, output_size, resize_with_pad)

  image = scale_input(image)

  return image

def preprocess_image(image,
                     output_size=224,
                     is_training=False,
                     bboxes=None,
                     use_square_crop=False,
                     resize_with_pad=False,
                     randaug_num_layers=None,
                     randaug_magnitude=None,
                     seed=None):
  if is_training:
    return preprocess_for_train(image,
                                output_size,
                                resize_with_pad,
                                bboxes,
                                use_square_crop,
                                randaug_num_layers,
                                randaug_magnitude,
                                seed=seed)
  else:
    return preprocess_for_eval(image,
                               output_size,
                               bboxes,
                               use_square_crop,
                               resize_with_pad)
