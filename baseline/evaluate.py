from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from PIL import Image

DATASET = '../../../dataset/Duke'
TEST = os.path.join(DATASET, 'bounding_box_test')
TEST_NUM = 17661
QUERY = os.path.join(DATASET, 'query')
QUERY_NUM = 2228

'''
DATASET = '../../../dataset/Market'
TEST = os.path.join(DATASET, 'bounding_box_test')
TEST_NUM = 19732
QUERY = os.path.join(DATASET, 'query')
QUERY_NUM = 3368

DATASET = '../../../dataset/CUHK03'
TEST = os.path.join(DATASET, 'bbox_test')
TEST_NUM = 5332
QUERY = os.path.join(DATASET, 'query')
QUERY_NUM = 1400
'''

def extract_feature(dir_path, net):
  features = []
  infos = []
  num = 0
  for image_name in os.listdir(dir_path):
    arr = image_name.split('_')
    person = int(arr[0])
    camera = int(arr[1][1])
    image_path = os.path.join(dir_path, image_name) 
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = net.predict(x)
    features.append(np.squeeze(feature))
    infos.append((person, camera))

  return features, infos

# use GPU to calculate the similarity matrix
query_t = tf.placeholder(tf.float32, (None, None))
test_t = tf.placeholder(tf.float32, (None, None))
query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


# load model
net = load_model('40.ckpt')
net = Model(input=net.input, output=net.get_layer('avg_pool').output)

test_f, test_info = extract_feature(TEST, net)
query_f, query_info = extract_feature(QUERY, net)

match = []
junk = []

for q_index, (qp, qc) in enumerate(query_info):
  tmp_match = []
  tmp_junk = []
  for t_index, (tp, tc) in enumerate(test_info):
    if tp == qp and qc != tc:
      tmp_match.append(t_index)
    elif tp == qp or tp == -1:
      tmp_junk.append(t_index)
  match.append(tmp_match)
  junk.append(tmp_junk)

result = sess.run(tensor, {query_t: query_f, test_t: test_f})
result_argsort = np.argsort(result, axis=1)

rank_1 = 0.0
mAP = 0.0

for idx in range(len(query_info)):
  recall = 0.0
  precision = 1.0
  hit = 0.0
  cnt = 0.0
  ap = 0.0
  YES = match[idx]
  IGNORE = junk[idx]
  rank_flag = True
  for i in list(reversed(range(0, TEST_NUM))):
    k = result_argsort[idx][i]
    if k in IGNORE:
      continue
    else:
      cnt += 1
      if k in YES:
        hit += 1
        if rank_flag:
          rank_1 += 1
      tmp_recall = hit/len(YES)
      tmp_precision = hit/cnt
      ap = ap + (tmp_recall - recall)*((precision + tmp_precision)/2)
      recall = tmp_recall
      precision = tmp_precision
      rank_flag = False
    if hit == len(YES):
      break
  mAP += ap

print ('Rank 1:\t%f'%(rank_1 / QUERY_NUM))
print ('mAP:\t%f'%(mAP / QUERY_NUM))
