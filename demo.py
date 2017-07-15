"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""

import warnings
from keras.applications import vgg16
from keras.models import Model
from keras import optimizers
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers.normalization import BatchNormalization

import random
import os
import h5py
import cv2
import numpy as np
from os.path import join
import argparse
import matplotlib.pyplot as plt


def load_im2(paths, img_cols, img_rows):
    '''Returns a list containing the loaded images from paths'''
    l = []

    if K.image_data_format() == 'channels_first': #theano
        for name in paths:
            #print(name)
            im2 = cv2.resize(cv2.imread(name), (img_cols, img_rows)).astype(np.float32)
            #print(im2.shape)
            # 'RGB'->'BGR'
            im2 = im2[::-1, :, :]
            # Zero-center by mean pixel
            im2 -= np.mean(im2)
            im2 = im2.transpose((2,0,1))
            l.append(im2)

    elif K.image_data_format() == 'channels_last': #tensorflow
        for name in paths:
            #print(name)
            im2 = cv2.resize(cv2.imread(name), (img_cols, img_rows)).astype(np.float32)
            # 'RGB'->'BGR'
            im2 = im2[:, :, ::-1]
            # Zero-center by mean pixel
            im2 -= np.mean(im2)
            l.append(im2)

    return l

def main():
    validation_data_dir = "dataset/val/"

    labels_path = "labels.txt"

    if os.path.exists(validation_data_dir + ".DS_Store"):
        os.remove(validation_data_dir + ".DS_Store")


    #set dimensions of the images
    img_rows, img_cols = 224, 224

    if K.image_data_format() == 'channels_first':
        shape_ord = (3, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 3)

    vgg16_model = vgg16.VGG16(weights=None, include_top=False, input_tensor=Input(shape_ord))

    #vgg16_model.summary()

    #add last fully-connected layers
    x = Flatten(input_shape=vgg16_model.output.shape)(vgg16_model.output)
    x = Dense(4096, activation='relu', name='ft_fc1')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(43, activation='softmax')(x)

    model = Model(inputs=vgg16_model.input, outputs=predictions)

    #compile the model
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                loss='categorical_crossentropy', metrics=['accuracy'])



    with open(labels_path, "r") as f:
        labels = f.readlines()

    old_validation_filenames = os.listdir(validation_data_dir)

    #select n_pic random pictures
    n_pic = 10
    validation_filenames = []
    for i in range(n_pic):
        validation_filenames.append(random.choice(old_validation_filenames))

    validation_filenames.sort()
    validation_images = []
    validation_labels = []

    for name in validation_filenames:
        if name.endswith(".ppm"):
            validation_images.append(validation_data_dir + name)
            label = name.split("_")[0]
            label_int = int(label)
            labels_array = [0]*43
            labels_array[label_int] = 1
            validation_labels.append(labels_array)
        else:
            validation_filenames.remove(name)

    print("Validation Filenames loaded.")


    validation = np.array(load_im2(validation_images, img_cols, img_rows))
    print("Validation images loaded.")

    model.load_weights("experiment_output/vgg16_deepstreet_training1.h5")

    predicted_labels = model.predict(validation)
    print("Labels predicted.")

    validation_labels_linear = []

    #make linear labels list
    for lbl in validation_labels:
        for i,val in enumerate(lbl):
            if val == 1:
                validation_labels_linear.append(i)


    predicted_labels = list(predicted_labels)

    for i in range(n_pic):
        img = validation[i]
        plt.imshow(img)
        print('Filename: {}'.format(validation_filenames[i]))
        print('Real label: {}'.format(labels[validation_labels_linear[i]]))
        print('Position of maximum probability: {}'.format(predicted_labels[i].argmax()))
        print('Maximum probability: {:.5f}'.format(max(predicted_labels[i])))
        print('Corresponding label: {}'.format(labels[predicted_labels[i].argmax()]))

        # sort top five predictions from softmax output
        top_inds = predicted_labels[i].argsort()[::-1][:5]  # reverse sort and take five largest items
        print('Top 5 probabilities and labels:')
        _ =[print('{:.5f} {}'.format(predicted_labels[i][k], labels[k])) for k in top_inds]
        print("\n\n")
        plt.show()



if __name__ == "__main__":
    main()
