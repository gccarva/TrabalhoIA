import imageisotropic
import imagewindoing
import glob
import cv2
import os 
basef = "rsna-breast-cancer-detection/"
folederimage = "train_images/"
foldertransform = "processed_images/"
print() 
for image in os.listdir(f"{basef}{folederimage}"):
    imagec  = cv2.imread(f"{basef}{folederimage}{image}")
    imagec = imagewindoing.windowimage(imagec)
    imagec = imageisotropic.isotropictimage(imagec)
    cv2.imwrite(f"{basef}{foldertransform}{image}",imagec)