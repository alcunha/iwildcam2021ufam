# Copyright 2021 Fagner Cunha
# Copyright 2021 TAO 2020 Winner Team
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Fagner Cunha to add support for iWildCam 2021

import os

import numpy as np
import onnx
import onnxruntime as rt

def change_input_dim(model):
  # Use some symbolic name not used for any other dimension
  sym_batch_dim = "N"
  # or an actal value
  actual_batch_dim = 4
  # The following code changes the first dimension of every input to be batch-dim
  # Modify as appropriate ... note that this requires all inputs to
  # have the same batch_dim
  model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim
  #model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim

def apply(transform, infile, outfile):
  model = onnx.load(infile)
  transform(model)
  onnx.save(model, outfile)
  return model

class ReID_Inference():
  def __init__(self,  path='reid_pytorch/reid1.onnx'):
    if os.path.exists(path):
      self.model = onnx.load(path)
    else:
      print('model not exists!')
      self.model = apply(change_input_dim, "reid1.onnx", path)
    #create runtime session
    self.sess = rt.InferenceSession(path)
    # get output name
    self.input_name = self.sess.get_inputs()[0].name
    print("input name", self.input_name)
    self.output_name= self.sess.get_outputs()[0].name
    print("output name", self.output_name)
    self.output_shape = self.sess.get_outputs()[0].shape
    print("output shape", self.output_shape)


  def forward_batch(self, batch_imgs):
    ress = self.sess.run([self.output_name], {self.input_name: batch_imgs})
    return np.array(ress)

  def __call__(self, imgs):
    features = self.forward_batch(imgs)
    features = features.reshape(imgs.shape[0], -1)
    return features
