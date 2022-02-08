from cProfile import label
import matplotlib.pyplot as plt
import glob
from keras.preprocessing import image
import os
import numpy as np

imagefiles = glob.glob("*.jpg")
zones = ["zone1","zone1-1", "zone2", "zone2-1", "zone3","zone3-1", "zone4","zone4-1", "zone5", "zone5-1"]
path = '/Users/yassinedehbi/work/myworkspace/finaldata/train/'
i = 0
fig = plt.figure(figsize=(6,6))
for zone in zones:  
    img =os.listdir(path+zone)[0]
    print(img)
    im = image.load_img(path+zone+'/'+img, target_size=(416, 416))
    fig.add_subplot(5, 2, zones.index(zone)+1, label=zone)
    plt.imshow(im)
plt.show()



            

        

