import tensorflow.keras.preprocessing.image as im
import glob
import os

def augmente_data():
    zones = ["zone1","zone1-1", "zone2", "zone2-1", "zone3","zone3-1", "zone4","zone4-1", "zone5", "zone5-1"]
    zones_check = ["zone1"]

    datagen = im.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False,
    featurewise_std_normalization=False, samplewise_std_normalization=False,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=40, width_shift_range=0.2,
    height_shift_range=0.2, brightness_range=None, shear_range=0.2, zoom_range=0.2,
    channel_shift_range=0.0, fill_mode='nearest',
    horizontal_flip=True, rescale=1./255,
    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None)
    
    parent_dir = '/Users/yassinedehbi/work/Stage/generated_im'
    path = '/Users/yassinedehbi/work/Stage/data/train/'
    images_files = glob.glob("*.JPG")
    for dir in zones:
        print(dir)
        path_target = os.path.join(parent_dir, dir)
        os.mkdir(path_target)
        for images_files in  glob.glob(path+dir+'/*.JPG'):
            print(images_files)
            img = im.load_img(images_files, target_size=(416, 416))
            img = im.img_to_array(img)
            img = img.reshape((1,) + img.shape)

            i = 0 
            for batch in datagen.flow(img, batch_size=1, save_to_dir=path_target, save_prefix='augm', save_format='jpg'):
                i+=1
                if i > 20:
                    break

augmente_data()