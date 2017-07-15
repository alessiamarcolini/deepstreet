import os
from shutil import copyfile
import random

t_target = 10
v_target = 2
train_path = "../dataset/train/"
val_path = "../dataset/val/"


new_train_path = "../small_dataset/train/"
new_val_path = "../small_dataset/val/"

if not os.path.exists("../small_dataset"):
    os.mkdir("../small_dataset")

if not os.path.exists(new_train_path):
    os.mkdir(new_train_path)

if not os.path.exists(new_val_path):
    os.mkdir(new_val_path)



train_filenames = os.listdir(train_path)
val_filenames = os.listdir(val_path)

t_classes = {}
v_classes = {}

for f in train_filenames:
    fsplit = f.split("_")
    cl = fsplit[0]
    if cl in t_classes:
        t_classes[cl].append(f)
    else:
        t_classes[cl] = []

for f in val_filenames:
    fsplit = f.split("_")
    cl = fsplit[0]
    if cl in v_classes:
        v_classes[cl].append(f)
    else:
        v_classes[cl] = []

#print(t_classes)
#print("\n\n")
#print(v_classes)

for j in sorted(t_classes):
    for i in range(t_target):
        #print(j,t_classes[j][0])
        print(j,i)
        f = random.choice(t_classes[j])
        copyfile(train_path + f, new_train_path + f)

for j in sorted(v_classes):
    for i in range(v_target):
        print(j,i)
        f = random.choice(v_classes[j])
        copyfile(val_path + f, new_val_path + f)
