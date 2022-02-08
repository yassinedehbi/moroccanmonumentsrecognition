import sys
import random
import math
import os
from matplotlib import pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Flatten, Input, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from os.path import isfile, join
from os import rename, listdir, rename, makedirs
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import glob
import re
import warnings
from keras.preprocessing import image
import tensorflow as tf
warnings.simplefilter(action='ignore', category=FutureWarning)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
import tensorflow.keras.preprocessing.image as im
from tqdm import tqdm



zones = ["zone1","zone1-1", "zone2", "zone2-1", "zone3","zone3-1", "zone4","zone4-1", "zone5", "zone5-1"]
zones_check = ["zone1"]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    train_path = '/Users/yassinedehbi/work/myworkspace/finaldata/train/'
    val_path = '/Users/yassinedehbi/work/myworkspace/finaldata/val/'
    test_path = '/Users/yassinedehbi/work/myworkspace/finaldata/test/'

    images_files = glob.glob("*.jpg")
    for dir in tqdm(zones):
        for images_files in  tqdm(glob.glob(train_path+dir+'/*.jpg')):
            #print(images_files)
            img = image.load_img(images_files, target_size=(416, 416))
            tr_x = image.img_to_array(img)
            tr_x = preprocess_input(tr_x)
            label = dir
            label_place = zones.index(label)
            x_train.append(tr_x)
            y_train.append(label_place)
    for dir in tqdm(zones):
        for images_files in  tqdm(glob.glob(val_path+dir+'/*.jpg')):
            #print(images_files)
            img = image.load_img(images_files, target_size=(416, 416))
            tr_x = image.img_to_array(img)
            tr_x = preprocess_input(tr_x)
            label = dir
            label_place = zones.index(label)
            x_val.append(tr_x)
            y_val.append(label_place)
    for dir in tqdm(zones):
        for images_files in  tqdm(glob.glob(test_path+dir+'/*.jpg')):
            #print(images_files)
            img = image.load_img(images_files, target_size=(416, 416))
            tr_x = image.img_to_array(img)
            tr_x = preprocess_input(tr_x)
            label = dir
            label_place = zones.index(label)
            x_test.append(tr_x)
            y_test.append(label_place)
    #print(type(x_train))
    #print(type(y_train))
    return np.array(x_train), np_utils.to_categorical(y_train), np.array(x_val), np_utils.to_categorical(y_val), np.array(x_test), np_utils.to_categorical(y_test)

print('###############################################################')
print('loanding data ...')
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data()
print('Data loaded')
def build_model(img_shape=(416, 416, 3), n_classes=10):
  
    base_model = InceptionResNetV2(include_top=False, weights='imagenet',
                             input_tensor=None, input_shape=img_shape)
    l = base_model.output
    l = AveragePooling2D((8, 8), strides=(8, 8))(l)
    l = Flatten(name='flatten')(l)
    l = Dense(512, activation='relu', kernel_initializer='he_uniform')(l)
    l = Dropout(0.25)(l)
    l = Dense(n_classes, activation='softmax', kernel_initializer='he_uniform')(l)

    model = Model(inputs=base_model.input, outputs=l)

    for layer in base_model.layers:
        layer.trainable = False
    
    adam = Adam(0.001) 
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  
    model.fit(X_train, Y_train,
                  batch_size=64,
                  epochs=20,
                  shuffle=True,
                  verbose=1,callbacks=[tensorboard_callback], validation_data=(X_val, Y_val)
                  )
    return model




model = build_model()

print("##########################################")
print('evaluating model')
score = model.evaluate(X_test, Y_test, verbose=1, batch_size= 64)

model.save('/Users/yassinedehbi/work/myworkspace/model/modell21.h5', save_format='h5')


print(score)