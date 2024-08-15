import torch 
import numpy as np
import cv2
import os 
import pydicom
from config import *
from torch.nn import functional as F
class PydicomMetadata:

    def __init__(self, ds):
        if "WindowWidth" not in ds or "WindowCenter" not in ds:
            self.window_widths = []
            self.window_centers = []
        else:
            ww = ds['WindowWidth']
            wc = ds['WindowCenter']
            self.window_widths = [float(e) for e in ww
                                  ] if ww.VM > 1 else [float(ww.value)]

            self.window_centers = [float(e) for e in wc
                                   ] if wc.VM > 1 else [float(wc.value)]

        # if nan --> LINEAR
        self.voilut_func = str(ds.get('VOILUTFunction', 'LINEAR')).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)


class DicomsdlMetadata:

    def __init__(self, ds):
        self.window_widths = ds.WindowWidth
        self.window_centers = ds.WindowCenter
        if self.window_widths is None or self.window_centers is None:
            self.window_widths = []
            self.window_centers = []
        else:
            try:
                if not isinstance(self.window_widths, list):
                    self.window_widths = [self.window_widths]
                self.window_widths = [float(e) for e in self.window_widths]
                if not isinstance(self.window_centers, list):
                    self.window_centers = [self.window_centers]
                self.window_centers = [float(e) for e in self.window_centers]
            except:
                self.window_widths = []
                self.window_centers = []

        # if nan --> LINEAR
        self.voilut_func = ds.VOILUTFunction
        if self.voilut_func is None:
            self.voilut_func = 'LINEAR'
        else:
            self.voilut_func = str(self.voilut_func).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)
# slow
# from pydicom's source
def _apply_windowing_np_v1(arr,
                           window_width=None,
                           window_center=None,
                           voi_func='LINEAR',
                           y_min=0,
                           y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.astype(np.float64)
    arr = arr.astype(np.float32)

    if voi_func in ['LINEAR', 'LINEAR_EXACT']:
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1
        below = arr <= (window_center - window_width / 2)
        above = arr > (window_center + window_width / 2)
        between = np.logical_and(~below, ~above)

        arr[below] = y_min
        arr[above] = y_max
        if between.any():
            arr[between] = ((
                (arr[between] - window_center) / window_width + 0.5) * y_range
                            + y_min)
    elif voi_func == 'SIGMOID':
        arr = y_range / (1 +
                         np.exp(-4 *
                                (arr - window_center) / window_width)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def _apply_windowing_np_v2(arr,
                           window_width=None,
                           window_center=None,
                           voi_func='LINEAR',
                           y_min=0,
                           y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.astype(np.float64)
    arr = arr.astype(np.float32)

    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1

        # simple trick to improve speed
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = np.clip(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        # simple trick to improve speed
        s = -4 / window_width
        arr = y_range / (1 + np.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def _apply_windowing_torch(arr,
                           window_width=None,
                           window_center=None,
                           voi_func='LINEAR',
                           y_min=0,
                           y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.double()
    arr = arr.float()

    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1

        # simple trick to improve speed
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = torch.clamp(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        # simple trick to improve speed
        s = -4 / window_width
        arr = y_range / (1 + torch.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def apply_windowing(arr,
                    window_width=None,
                    window_center=None,
                    voi_func='LINEAR',
                    y_min=0,
                    y_max=255,
                    backend='np_v2'):
    if backend == 'torch':
        if isinstance(arr, torch.Tensor):
            pass
        elif isinstance(arr, np.ndarray):
            if arr.dtype == np.uint16:
                arr = torch.from_numpy(arr, torch.int16)
            else:
                arr = torch.from_numpy(arr)

    if backend == 'np_v1':
        windowing_func = _apply_windowing_np_v1
    elif backend == 'np_v2':
        windowing_func = _apply_windowing_np_v2
    elif backend == 'torch':
        windowing_func = _apply_windowing_torch
    else:
        raise ValueError(
            f'Invalid backend {backend}, must be one of ["np", "np_v2", "torch"]'
        )

    arr = windowing_func(arr,
                         window_width=window_width,
                         window_center=window_center,
                         voi_func=voi_func,
                         y_min=y_min,
                         y_max=y_max)
    return arr
def min_max_scale(img):
    maxv = img.max()
    minv = img.min()
    if maxv > minv:
        return (img - minv) / (maxv - minv)
    else:
        return img - minv  # ==0


#@TODO: percentile on both min-max?
# this version is not correctly implemented, but used in the winning submission
def percentile_min_max_scale(img, pct=99):
    if isinstance(img, np.ndarray):
        maxv = np.percentile(img, pct) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv  # ==0
        ret = np.clip(ret, 0, 1)
    elif isinstance(img, torch.Tensor):
        maxv = torch.quantile(img, pct / 100) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv  # ==0
        ret = torch.clamp(ret, 0, 1)
    else:
        raise ValueError(
            'Invalid img type, should be numpy array or torch.Tensor')
    return ret


def resize_and_pad(img, input_size=MODEL_INPUT_SIZE):
    input_h, input_w = input_size
    ori_h, ori_w = img.shape[:2]
    ratio = min(input_h / ori_h, input_w / ori_w)
    # resize
    img = F.interpolate(img.view(1, 1, ori_h, ori_w),
                        mode="bilinear",
                        scale_factor=ratio,
                        recompute_scale_factor=True)[0, 0]
    # padding
    padded_img = torch.zeros((input_h, input_w),
                             dtype=img.dtype,
                             device='cuda')
    cur_h, cur_w = img.shape
    y_start = (input_h - cur_h) // 2
    x_start = (input_w - cur_w) // 2
    padded_img[y_start:y_start + cur_h, x_start:x_start + cur_w] = img
    padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
    return padded_img


def save_img_to_file(save_path, img, backend='cv2'):
    file_ext = os.path.basename(save_path).split('.')[-1]
    if backend == 'cv2':
        if img.dtype == np.uint16:
            # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
            assert file_ext in ['png', 'jp2', 'tiff', 'tif']
            cv2.imwrite(save_path, img)
        elif img.dtype == np.uint8:
            cv2.imwrite(save_path, img)
        else:
            raise ValueError(
                '`cv2` backend only support uint8 or uint16 images.')
    elif backend == 'np':
        assert file_ext == 'npy'
        np.save(save_path, img)
    else:
        raise ValueError(f'Unsupported backend `{backend}`.')


def load_img_from_file(img_path, backend='cv2'):
    if backend == 'cv2':
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    elif backend == 'np':
        return np.load(img_path)
    else:
        raise ValueError()
        

def make_uid_transfer_dict(df, dcm_root_dir):
    machine_id_to_transfer = {}
    machine_id = df.machine_id.unique()
    for i in machine_id:
        row = df[df.machine_id == i].iloc[0]
        sample_dcm_path = os.path.join(dcm_root_dir, str(row.patient_id),
                                       f'{row.image_id}.dcm')
        dicom = pydicom.dcmread(sample_dcm_path)
        machine_id_to_transfer[i] = dicom.file_meta.TransferSyntaxUID
    return machine_id_to_transfer