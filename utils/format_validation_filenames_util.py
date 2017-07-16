import os

with open("validation_classes.csv", "r") as f:
    rows = f.readlines()

rows = rows[1:-1]

rows = [x for x in rows if x != "\n"]

path = "dataset/val/"

for row in rows:
    rsplit = row.split(";")
    filename = rsplit[0]
    c = int(rsplit[1])

    new_filename = format(c,'05d') + "_" + filename

    if os.path.exists(path + filename):
        os.rename(path + filename, path + new_filename)
