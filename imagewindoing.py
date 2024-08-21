import cv2
import torch
import numpy as np 
from windowing import apply_windowing
# Load the image as a grayscale image (single channel)
for i in range(2):
    image_path = f'teste{i+1}.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean_value = np.mean(image)
    # Convert the image to a PyTorch tensor
    image_torch = torch.tensor(image, dtype=torch.float32)

    # Apply windowing using PyTorch backend
    windowed_image_torch = apply_windowing(
        image_torch,
        window_width=100,       # Example window width
        window_center=mean_value*1.3,      # Example window center
        voi_func='LINEAR',      # Windowing function
        y_min=0,                # Minimum intensity value
        y_max=255,              # Maximum intensity value
        backend='torch'         # Use the PyTorch backend
    )

    # Convert back to NumPy for saving or display (if needed)
    windowed_image_np = windowed_image_torch.numpy().astype(np.uint8)

    # Save or display the windowed image
    cv2.imwrite(f'teste{i+1}w.png', windowed_image_np)

