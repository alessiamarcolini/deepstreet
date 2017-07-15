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

import os
import h5py
import cv2
import numpy as np
from os.path import join
from mcc_multiclass import multimcc, confusion_matrix
import argparse
import matplotlib.pyplot as plt
from timeit import default_timer as timer

start = timer()

class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg


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
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    parser = myArgumentParser(description='Run a prediction experiment using pretrained VGG16, specified on the deepstreet DataSet.',
            fromfile_prefix_chars='@')
    parser.add_argument('--gpu', type=int, default=0, help='GPU Device (default: %(default)s)')
    parser.add_argument('--output_dir', type=str, default="./experiment_output/",help='Output directory')
    parser.add_argument('--input_dir', type=str, default="./",help='Input directory')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

    args = parser.parse_args()
    GPU = args.gpu
    OUTDIR = args.output_dir+"/"
    INDIR = args.input_dir+"/"
    DEBUG = args.debug

    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)


    if DEBUG:
        validation_data_dir = INDIR + "small_dataset/val/"
    else:
        #validation_data_dir = "dataset/val/"
        validation_data_dir = INDIR + "val/"

    if os.path.exists(INDIR + validation_data_dir + ".DS_Store"):
        os.remove(INDIR + validation_data_dir + ".DS_Store")

    #set dimensions of the images
    img_rows, img_cols = 224, 224

    if K.image_data_format() == 'channels_first':
        shape_ord = (3, img_rows, img_cols)
    else:  # channel_last
        shape_ord = (img_rows, img_cols, 3)

    #load pre-trained VGG16 with ImageNet weights
    vgg16_model = vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape_ord))
    vgg16_model.summary()

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


    #load validation images and create labels list
    validation_filenames = os.listdir(validation_data_dir)
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

    #write summary file
    prediction_summary = open(OUTDIR + "vgg16_deepstreet_t_prediction_summary_deepstreet_v.txt", "w")
    prediction_summary.write("\t".join(['FILENAME', 'REAL_LABEL', 'PREDICTED_LABELS']) + '\n')

    predicted_labels_linear = []
    validation_labels_linear = []

    #make linear labels list
    for lbl in validation_labels:
        for i,val in enumerate(lbl):
            if val == 1:
                validation_labels_linear.append(i)


    for i in range(len(predicted_labels)):
        cls_prob = predicted_labels[i]     #percentage of belonging for i image

        predicted_label_index = np.argmax(cls_prob) #get the index of the class with higher probability
        line = [validation_images[i], str(validation_labels_linear[i]), str(predicted_label_index), str(round(cls_prob[predicted_label_index],3))]

        s = ""
        for i in range(42):
            s += "{}:{}; ".format(i,round(cls_prob[i],3))
            #s += str(i) + ":" + str(round(cls_prob[i],3)) + "; "
        s += "42:{}".format(round(cls_prob[42],3))
        #s += "42:" + str(round(cls_prob[42],3))

        line.append(s)

        predicted_labels_linear.append(np.argmax(cls_prob))
        prediction_summary.write(";".join(line) + "\n")
        prediction_summary.flush()


    validation_labels_linear = np.array(validation_labels_linear)
    predicted_labels_linear = np.array(predicted_labels_linear)

    #calculate MCC
    MCC = multimcc(validation_labels_linear, predicted_labels_linear)
    print(MCC)

    prediction_summary.write("MCC = {}".format(MCC))
    prediction_summary.flush()
    prediction_summary.close()

    #compute confusion matrix and save the image
    conf_matrix = confusion_matrix(validation_labels_linear,predicted_labels_linear)[0]
    plt.matshow(conf_matrix)
    plt.colorbar()
    plt.savefig("confusion_matrix.png")

    end = timer()
    print("Total time: ", end - start)

if __name__ == "__main__":
    main()
