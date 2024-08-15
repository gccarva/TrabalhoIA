BATCH_SIZE = 2
# binarization threshold for classification
THRES = 0.31
AUTO_THRES = False
AUTO_THRES_PERCENTILE = 0.97935

# classification model
USE_TRT = True

J2K_SUID = '1.2.840.10008.1.2.4.90'
J2K_HEADER = b"\x00\x00\x00\x0C"
JLL_SUID = '1.2.840.10008.1.2.4.70'
JLL_HEADER = b"\xff\xd8\xff\xe0"
SUID2HEADER = {J2K_SUID: J2K_HEADER, JLL_SUID: JLL_HEADER}
VOILUT_FUNCS_MAP = {'LINEAR': 0, 'LINEAR_EXACT': 1, 'SIGMOID': 2}
VOILUT_FUNCS_INV_MAP = {v: k for k, v in VOILUT_FUNCS_MAP.items()}
# roi detection
ROI_YOLOX_INPUT_SIZE = [416, 416]
ROI_YOLOX_CONF_THRES = 0.5
ROI_YOLOX_NMS_THRES = 0.9
ROI_YOLOX_HW = [(52, 52), (26, 26), (13, 13)]
ROI_YOLOX_STRIDES = [8, 16, 32]
ROI_AREA_PCT_THRES = 0.04

# model
MODEL_INPUT_SIZE = [2048, 1024]

MODE = 'KAGGLE-TEST'
assert MODE in ['LOCAL-VAL', 'KAGGLE-VAL', 'KAGGLE-TEST']

# settings corresponding to each mode
if MODE == 'KAGGLE-VAL':
    TRT_MODEL_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_ensemble_batch2_fp32_torch2trt.engine'
    TORCH_MODEL_CKPT_PATHS = [
        f'/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_fold_{i}.pth.tar'
        for i in range(4)
    ]
    ROI_YOLOX_ENGINE_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/yolox_nano_416_roi_trt_p100.pth'
    CSV_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/_val_fold_0.csv'
    DCM_ROOT_DIR = '/kaggle/input/rsna-breast-cancer-detection/train_images'
    SAVE_IMG_ROOT_DIR = '/kaggle/tmp/pngs'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = False
elif MODE == 'KAGGLE-TEST':
    TRT_MODEL_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_ensemble_batch2_fp32_torch2trt.engine'
    TORCH_MODEL_CKPT_PATHS = [
        f'/kaggle/input/rsna-breast-cancer-detection-best-ckpts/best_convnext_fold_{i}.pth.tar'
        for i in range(4)
    ]
    ROI_YOLOX_ENGINE_PATH = '/kaggle/input/rsna-breast-cancer-detection-best-ckpts/yolox_nano_416_roi_trt_p100.pth'
    CSV_PATH = '/kaggle/input/rsna-breast-cancer-detection/test.csv'
    DCM_ROOT_DIR = '/kaggle/input/rsna-breast-cancer-detection/test_images'
    SAVE_IMG_ROOT_DIR = '/kaggle/tmp/pngs'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = True
elif MODE == 'LOCAL-VAL':
    TRT_MODEL_PATH = './assets/best_convnext_ensemble_batch2_fp32_torch2trt.engine'
    TORCH_MODEL_CKPT_PATHS = [
        f'./assets/best_convnext_fold_{i}.pth.tar'
        for i in range(4)
    ]
    ROI_YOLOX_ENGINE_PATH = '../roi_det/YOLOX/YOLOX_outputs/yolox_nano_bre_416/model_trt.pth'
    CSV_PATH = '../../datasets/cv/v1/val_fold_0.csv'
    DCM_ROOT_DIR = '../../datasets/train_images/'
    SAVE_IMG_ROOT_DIR = './temp_save'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = False