import os
import random

path = "../dataset/train/"

l_images = os.listdir(path)

classes = {}

target = 1 #number of images to remove

for i in l_images:
    isplit = i.split("_")
    cl = isplit[0]
    if cl in classes:
        classes[cl].append(i)
    else:
        classes[cl] = []


for i in sorted(classes):
    for k in range(target):
        im = random.choice(classes[i])
        classes[i].remove(im)
        os.remove(path + im)
