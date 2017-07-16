import os

images = os.listdir("../dataset/train/")

classes = {}

for i in images:
    isplit = i.split("_")
    cl = isplit[0]
    if cl in classes:
        classes[cl] += 1
    else:
        classes[cl] = 1


for c in sorted(classes):
    print(c,classes[c])
