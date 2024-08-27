import os
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from yolov8_cam.eigen_cam import EigenCAM
from yolov8_cam.utils.image import show_cam_on_image
model = YOLO("yolov8m.pt")
img = cv2.imread('images/bird-dog-cat.jpg')
rgb_img = img.copy()
# Select what principal components you want highlighted, zero-indexed
target_layers = [model.model.model[-4]]
principal_comp = [0,1]
cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(
    rgb_img,
    eigen_smooth=True,
    principal_comp=principal_comp,
)

for i in range(grayscale_cam.shape[3]):
    cam_image = show_cam_on_image(img, grayscale_cam[0,:,:,i], use_rgb=True)
    print(f"principal component {principal_comp[i]+1}")
    plt.imshow(cam_image)
    plt.show()
