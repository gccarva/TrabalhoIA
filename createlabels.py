#! /usr/bin/python3
# from ultralytics.yolo.data.converter import convert_coco
# convert_coco("./train_images", cls91to80=False)
from pylabel import importer
from pathlib import Path
import os
dataset = importer.ImportCoco(
    path="./yolo_labels/labels.json", path_to_images="./yolo_labels/test/images")
dataset.export.ExportToYoloV5(output_path="./yolo_labels/test/labels")
pathlist = Path("./yolo_labels/train/labels").glob('**/*.txt')
for path in pathlist:
    # because path is object not string
   # print(path)
    path_in_str = str(path)
    linhas = set()
    with open(path_in_str) as file:
        linhas = set(file.readlines())
    with open(path_in_str, "w") as file:
        for linha in linhas:
            file.write(linha)
os.remove("./yolo_labels/test/dataset.yaml")
