import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

import numpy as np
import matplotlib.image as mpimg
import os
import random

def augment_brightness_camera_images(image):

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


def main():
    PATH = "dataset/train"

    if os.path.exists(PATH + "/.DS_Store"):
        os.remove(PATH + "/.DS_Store")

    l_images = os.listdir(PATH)


    target = 1500
    classes = {}
    for i in l_images:
        isplit = i.split("_")
        cl = isplit[0]
        if cl in classes:
            classes[cl].append(i)
        else:
            classes[cl] = []


    for k, c in enumerate(sorted(classes)):
        cont = 0
        remaining = target - len(classes[c])
        for i in range(remaining):
            filename = random.choice(classes[c])
            image = cv2.imread(PATH + "/" + filename,1)
            img = transform_image(image,20,10,1,brightness=1)
            scont = format(cont,'05d')
            cv2.imwrite("img/%s_000tr_%s_t.ppm" % (c,scont) , img)
            cont += 1


if __name__ == "__main__":
    main()
