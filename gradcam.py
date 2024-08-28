import os
from zipfile import ZipFile
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
np.object = object
from ultralytics import YOLO
from yolov8_cam.eigen_cam import EigenCAM
from yolov8_cam.utils.image import show_cam_on_image

model = YOLO("/home/fernando/Documents/ia/runs/detect/medium treinado/weights/best.pt")
img = cv2.imread('/home/fernando/Documents/ia/train_images/images/10208@74874399.png')


img_normalized = img.astype(float) / 255
rgb_img = img_normalized.copy()

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
    cam_image = show_cam_on_image(img_normalized, grayscale_cam[0,:,:,i], use_rgb=True)
    cam_image = np.transpose(cam_image,(1,0,2))
    cv2.imwrite(f"mapacalor{i}.png",cam_image)
    print(f"principal component {principal_comp[i]+1}")
    plt.imshow(cam_image)
    plt.show()
