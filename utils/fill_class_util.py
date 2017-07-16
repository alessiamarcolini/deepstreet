import os

source_path = "img/" #image augmentation directory
dest_path = "../dataset/train/"

s_im = os.listdir(source_path)
d_im = os.listdir(dest_path)

classes = {}

for i in d_im:
    isplit = i.split("_")
    cl = isplit[0]
    if cl in classes:
        classes[cl] += 1
    else:
        classes[cl] = 1

for cl in sorted(classes):
    remaining = 1501 - classes[cl]
    for i in range(remaining):
        for im in s_im:
            c = im.split("_")[0]
            if cl == c and not os.path.exists(dest_path + im):
                #print("si, ",im)
                os.rename(source_path + im, dest_path + im)
