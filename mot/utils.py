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

import calendar
import datetime
import re

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'datetime_format', default='%Y-%m-%d %H:%M:%S.000',
    help=('Datetime format used to convert to days to float'))

def xor(a, b):
  return bool(a) ^ bool(b)

def deg2rad(deg):
  pi_on_180 = 0.017453292519943295
  return deg * pi_on_180

def get_valid_filename(file_name):
  file_name = re.sub(r'[()]', '', file_name)
  file_name = re.sub(r'\s', '-', file_name)

  return file_name

def is_number(text):
  try:
    float(text)
    return True
  except ValueError:
    return False

def date2float(date):
  dt = datetime.datetime.strptime(date, FLAGS.datetime_format).timetuple()
  year_days = 366 if calendar.isleap(dt.tm_year) else 365

  return dt.tm_yday/year_days
