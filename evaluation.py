import os
from PIL.Image import ROTATE_90
from tensorflow import keras
import glob
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.python.ops.gen_math_ops import arg_max


zones = ["zone1","zone1-1", "zone2", "zone2-1", "zone3","zone3-1", "zone4","zone4-1", "zone5", "zone5-1"]
zones_check = ["zone1"]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():
    imgs = []
    x_test = []
    y_test = []
    true_labels = []
    train_path = '/Users/yassinedehbi/work/myworkspace//finaldata/train/'
    val_path = '/Users/yassinedehbi/work/myworkspace//finaldata/val/'
    test_path = '/Users/yassinedehbi/work/myworkspace//finaldata/test/'

    images_files = glob.glob("*.JPG")
    for dir in tqdm(zones):
        for images_files in  tqdm(glob.glob(test_path+dir+'/*.jpg')):
            #print(images_files)
            img = image.load_img(images_files, target_size=(416, 416))
            imgs.append(img)
            tr_x = image.img_to_array(img)
            tr_x = preprocess_input(tr_x)
            label = dir
            label_place = zones.index(label)
            true_labels.append(label_place)
            x_test.append(tr_x)
            y_test.append(label_place)
   
    return np.array(x_test), np_utils.to_categorical(y_test),true_labels, imgs

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X_test, Y_test, true_label, imgs = load_data()

print("############################# DATA LOADED ##############################")

model = keras.models.load_model('/Users/yassinedehbi/work/myworkspace/model/modell21.h5')
print("############################# MODEL LOADED ##############################")
predictions = model.predict(X_test)
predd = np.argmax(predictions, axis=1) 
conf_matrix = tf.math.confusion_matrix(true_label, predd)
print(conf_matrix.numpy())
df_cm = pd.DataFrame(conf_matrix.numpy(), zones, zones)
mapping = { i : zones[i] for i in range(10)}
sns.set(font_scale=1.4) 
sns.heatmap(df_cm, annot=True) 
plt.show()


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i] 
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(zones[predicted_label],
                                100*np.max(predictions_array),
                                zones[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(zones, predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



for i in range(20):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], true_label, imgs)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  true_label)
    plt.show()
    i+= 200



