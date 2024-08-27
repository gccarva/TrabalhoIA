import numpy as np
np.object = object
from ultralytics import YOLO
 
# tentar descobrir um bom imgsz e batch
"""YOLOv9t	640	38.3	53.1	2.0	7.7
YOLOv9s	640	46.8	63.4	7.2	26.7
YOLOv9m	640	51.4	68.1	20.1	76.8
YOLOv9c	640	53.0	70.2	25.5	102.8
YOLOv9e	640	55.6	72.8	58.1	192.5"""
#https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset
new = 1
if (new == 1):
    model = YOLO('yolov8s.pt')
else:
    model = YOLO("runs/detect/train/weights/last.pt")
model.train(data="./dataset_yolo.yaml", epochs=100,batch=2,device="0")
