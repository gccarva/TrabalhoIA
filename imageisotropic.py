import numpy as np
np.object =object
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision.transforms.functional as TF

# Define the custom resize transformation
class CustomResize(A.ImageOnlyTransform):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, image, **params):
        return cv2.resize(image, (self.width, self.height))

# Define the dimensions
max_h = 1024*2
max_w = 1024

# Define the transformation pipeline
transform_fn = A.Compose([
    CustomResize(height=max_h, width=max_w, always_apply=True, p=1.0),
    A.PadIfNeeded(min_height=max_h,
                  min_width=max_w,
                  position=A.augmentations.geometric.transforms.PadIfNeeded.PositionType.CENTER,
                  border_mode=cv2.BORDER_CONSTANT,
                  value=0,
                  always_apply=True,
                  p=1.0),
    #ToTensorV2(transpose_mask=True)
])

# Load an image (for example, from a file)
image_path = 'teste1w.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale if needed

# Apply the transformations
transformed = transform_fn(image=image)
transformed_image_tensor = transformed['image']

# Convert tensor to PIL image


cv2.imwrite('teste1seila.png', transformed_image_tensor)
# Save or display the transformed image

