from __future__ import division, print_function, absolute_import

import os
import numpy as np

import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout
from keras.initializers import RandomNormal
from keras.models import Model
from keras import backend as K
from keras.models import load_model

DATASET = '../dataset/Duke'
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bounding_box_train')

'''
DATASET = '../dataset/Market'
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bounding_box_train')

DATASET = '../dataset/CUHK03'
LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bbox_train')
'''

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)

# load pre-trained resnet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

x = base_model.output
x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
x = Dense(702, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
net = Model(input=base_model.input, output=x)

for layer in net.layers:
   layer.trainable = True

# load data
images, labels = [], []
with open(LIST, 'r') as f:
  for line in f:
    line = line.strip()
    img, lbl = line.split()
    img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img) 

    images.append(img[0])
    labels.append(int(lbl))

images = np.array(images)
labels = to_categorical(labels)

# train
batch_size = 16
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20, # 0. 
    width_shift_range=0.2, # 0.
    height_shift_range=0.2,# 0.
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    data_format=K.image_data_format())

net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')
net.fit_generator(datagen.flow(images, labels, batch_size=batch_size), steps_per_epoch=len(images)/batch_size+1, epochs=40)
net.save('0.ckpt') 
