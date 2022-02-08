from tensorflow import keras
import sys
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing import image
import numpy as np

IMG_PATH = sys.argv[1]
print(IMG_PATH)

zones = ["zone1","zone1-1", "zone2", "zone2-1", "zone3","zone3-1", "zone4","zone4-1", "zone5", "zone5-1"]
MODEL_PATH = "/Users/yassinedehbi/work/Stage/model/modell20.h5"
model = keras.models.load_model(MODEL_PATH)

myimg = preprocess_input(image.img_to_array(image.load_img(IMG_PATH, target_size=(416,416,3))))
img_list = []
img_list.append(np.array(myimg))
pred = model.predict(np.asarray(img_list))

prediction = zones[np.argmax(pred)]

print(prediction)





