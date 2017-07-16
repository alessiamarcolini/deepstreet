import os

with open("../validation_info.csv", "r") as f:
    rows = f.readlines()


sout = "filename;class_id\n"

for r in rows:
    ssplit = r.split(";")
    name = ssplit[0]
    nsplit = name.split(".")
    print(nsplit)
    name = nsplit[0] + "_v." + nsplit[1]
    sout += "{};{}\n".format(name, ssplit[7])

with open("../validation_classes.csv", "w") as f1:
    f1.write(sout)
