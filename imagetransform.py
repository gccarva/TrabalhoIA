import cv2
import albumentations as A
import augs  # Assuming your custom augmentation is in this module

# Define the transformation pipeline
transform_fn = A.Compose(
            [
                # crop
                augs.CustomRandomSizedCropNoResize(scale=(0.5, 1.0),
                                                   ratio=(0.5, 0.8),
                                                   always_apply=False,
                                                   p=0.4),

                # flip
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # downscale
                A.OneOf([
                    A.Downscale(scale_min=0.75,
                                scale_max=0.95,
                                interpolation=dict(upscale=cv2.INTER_LINEAR,
                                                   downscale=cv2.INTER_AREA),
                                always_apply=False,
                                p=0.1),
                    A.Downscale(scale_min=0.75,
                                scale_max=0.95,
                                interpolation=dict(upscale=cv2.INTER_LANCZOS4,
                                                   downscale=cv2.INTER_AREA),
                                always_apply=False,
                                p=0.1),
                    A.Downscale(scale_min=0.75,
                                scale_max=0.95,
                                interpolation=dict(upscale=cv2.INTER_LINEAR,
                                                   downscale=cv2.INTER_LINEAR),
                                always_apply=False,
                                p=0.8),
                ],
                        p=0.125),

                # contrast
                # relative dark/bright between region, like HDR
                A.OneOf([
                    A.RandomToneCurve(scale=0.3, always_apply=False, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                                               contrast_limit=(-0.4, 0.5),
                                               brightness_by_max=True,
                                               always_apply=False,
                                               p=0.5)
                ],
                        p=0.5),

                # affine
                A.OneOf(
                    [
                        A.ShiftScaleRotate(shift_limit=None,
                                           scale_limit=[-0.15, 0.15],
                                           rotate_limit=[-30, 30],
                                           interpolation=cv2.INTER_LINEAR,
                                           border_mode=cv2.BORDER_CONSTANT,
                                           value=0,
                                           mask_value=None,
                                           shift_limit_x=[-0.1, 0.1],
                                           shift_limit_y=[-0.2, 0.2],
                                           rotate_method='largest_box',
                                           always_apply=False,
                                           p=0.6),

                        # one of with other affine
                        A.ElasticTransform(alpha=1,
                                           sigma=20,
                                           alpha_affine=10,
                                           interpolation=cv2.INTER_LINEAR,
                                           border_mode=cv2.BORDER_CONSTANT,
                                           value=0,
                                           mask_value=None,
                                           approximate=False,
                                           same_dxdy=False,
                                           always_apply=False,
                                           p=0.2),

                        # distort
                        A.GridDistortion(num_steps=5,
                                         distort_limit=0.3,
                                         interpolation=cv2.INTER_LINEAR,
                                         border_mode=cv2.BORDER_CONSTANT,
                                         value=0,
                                         mask_value=None,
                                         normalized=True,
                                         always_apply=False,
                                         p=0.2),
                    ],
                    p=0.5),

                # random erase
                A.CoarseDropout(max_holes=6,
                                max_height=0.15,
                                max_width=0.25,
                                min_holes=1,
                                min_height=0.05,
                                min_width=0.1,
                                fill_value=0,
                                mask_fill_value=None,
                                always_apply=False,
                                p=0.25),
            ],
            p=0.9)

# Load the image#
image_path = 'teste.png'
#image = cv2.imread(image_path)
def transformimage(image):
# Apply the transformation
    transformed = transform_fn(image=image)
    transformed_image = transformed['image']
    return transformed_image
# Save or display the image

