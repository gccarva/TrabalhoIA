import gc
import multiprocessing
import os
import shutil
import time

import cv2
import dicomsdl
import numpy as np
import nvidia.dali as dali
import pandas as pd
import pydicom
import torch
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut, pixel_dtype
from tqdm import tqdm

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import ctypes

import helpers as misc_utils
import nvidia.dali.types as types
from nvidia.dali import types
from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.experimental import eager
from nvidia.dali.types import DALIDataType
from torch.nn import functional as F
from tqdm import tqdm
from config import *
import roi_extract
from helpers import apply_windowing,min_max_scale,resize_and_pad,save_img_to_file
# DALI patch for INT16 support
################################################################################
DALI2TORCH_TYPES = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.UINT16: torch.int16,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64
}

TORCH_DTYPES = {
    'uint8': torch.uint8,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}


# @TODO: dangerous to copy from UINT16 to INT16 (memory layout?)
# little/big endian ?
# @TODO: faster reuse memory without copying: https://github.com/NVIDIA/DALI/issues/4126
def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = DALI2TORCH_TYPES[dali_tensor.dtype]

    assert dali_type == arr.dtype, (
        "The element type of DALI Tensor/TensorList"
        " doesn't match the element type of the target PyTorch Tensor: "
        "{} vs {}".format(dali_type, arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
        ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
            format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


class _JStreamExternalSource:
    """DALI External Source for in-memory dicom decoding"""

    def __init__(self, dcm_paths, batch_size=1):
        self.dcm_paths = dcm_paths
        self.len = len(dcm_paths)
        self.batch_size = batch_size

    def __call__(self, batch_info):
        idx = batch_info.iteration
        # print('IDX:', batch_info.iteration, batch_info.epoch_idx)
        start = idx * self.batch_size
        end = min(self.len, start + self.batch_size)
        if end <= start:
            raise StopIteration()

        batch_dcm_paths = self.dcm_paths[start:end]
        j_streams = []
        inverts = []
        windowing_params = []
        voilut_funcs = []

        for dcm_path in batch_dcm_paths:
            ds = pydicom.dcmread(dcm_path)
            pixel_data = ds.PixelData
            offset = pixel_data.find(
                SUID2HEADER[ds.file_meta.TransferSyntaxUID])
            j_stream = np.array(bytearray(pixel_data[offset:]), np.uint8)
            invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
            meta = PydicomMetadata(ds)
            windowing_param = np.array(
                [meta.window_centers, meta.window_widths], np.float16)
            voilut_func = VOILUT_FUNCS_MAP[meta.voilut_func]
            j_streams.append(j_stream)
            inverts.append(invert)
            windowing_params.append(windowing_param)
            voilut_funcs.append(voilut_func)
        return j_streams, np.array(inverts, dtype=np.bool_), \
            windowing_params, np.array(voilut_funcs, dtype=np.uint8)


@dali.pipeline_def
def _dali_pipeline(eii):
    jpeg, invert, windowing_param, voilut_func = dali.fn.external_source(
        source=eii,
        num_outputs=4,
        dtype=[
            dali.types.UINT8, dali.types.BOOL, dali.types.FLOAT16,
            dali.types.UINT8
        ],
        batch=True,
        batch_info=True,
        parallel=True)
    ori_img = dali.fn.experimental.decoders.image(
        jpeg,
        device='mixed',
        output_type=dali.types.ANY_DATA,
        dtype=dali.types.UINT16)
    return ori_img, invert, windowing_param, voilut_func


def decode_crop_save_dali(roi_yolox_engine_path,
                          dcm_paths,
                          save_paths,
                          save_backend='cv2',
                          batch_size=1,
                          num_threads=1,
                          py_num_workers=1,
                          py_start_method='fork',
                          device_id=0):
    """DALI dicom decoding --> ROI cropping --> norm --> save as 8-bits PNG"""
    
    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']
    num_dcms = len(dcm_paths)

    # dali to process with chunk in-memory
    external_source = _JStreamExternalSource(dcm_paths, batch_size=batch_size)
    pipe = _dali_pipeline(
        external_source,
        py_num_workers=py_num_workers,
        py_start_method=py_start_method,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        debug=False,
    )
    pipe.build()

    roi_extractor = roi_extract.RoiExtractor(engine_path=roi_yolox_engine_path,
                                             input_size=ROI_YOLOX_INPUT_SIZE,
                                             num_classes=1,
                                             conf_thres=ROI_YOLOX_CONF_THRES,
                                             nms_thres=ROI_YOLOX_NMS_THRES,
                                             class_agnostic=False,
                                             area_pct_thres=ROI_AREA_PCT_THRES,
                                             hw=ROI_YOLOX_HW,
                                             strides=ROI_YOLOX_STRIDES,
                                             exp=None)
    print('ROI extractor (YOLOX) loaded!')

    num_batchs = num_dcms // batch_size
    last_batch_size = batch_size
    if num_dcms % batch_size > 0:
        num_batchs += 1
        last_batch_size = num_dcms % batch_size

    cur_idx = -1
    for _batch_idx in tqdm(range(num_batchs)):
        try:
            outs = pipe.run()
        except Exception as e:
            #             print('DALI exception occur:', e)
            print(
                f'Exception: One of {dcm_paths[_batch_idx * batch_size: (_batch_idx + 1) * batch_size]} can not be decoded.'
            )
            # ignore this batch and re-build pipeline
            if _batch_idx < num_batchs - 1:
                cur_idx += batch_size
                del external_source, pipe
                gc.collect()
                torch.cuda.empty_cache()
                external_source = _JStreamExternalSource(
                    dcm_paths[(_batch_idx + 1) * batch_size:],
                    batch_size=batch_size)
                pipe = _dali_pipeline(
                    external_source,
                    py_num_workers=py_num_workers,
                    py_start_method=py_start_method,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=device_id,
                    debug=False,
                )
                pipe.build()
            else:
                cur_idx += last_batch_size
            continue

        imgs = outs[0]
        inverts = outs[1]
        windowing_params = outs[2]
        voilut_funcs = outs[3]
        for j in range(len(inverts)):
            cur_idx += 1
            save_path = save_paths[cur_idx]
            img_dali = imgs[j]
            img_torch = torch.empty(img_dali.shape(),
                                    dtype=torch.int16,
                                    device='cuda')
            feed_ndarray(img_dali,
                         img_torch,
                         cuda_stream=torch.cuda.current_stream(device=0))
            # @TODO: test whether copy uint16 to int16 pointer is safe in this case
            if 0:
                img_np = img_dali.as_cpu().squeeze(-1)  # uint16
                print(type(img_np), img_np.shape)
                img_np = torch.from_numpy(img_np, dtype=torch.int16)
                diff = torch.max(torch.abs(img_np - img_torch))
                assert diff == 0, f'{img_torch.shape}, {img_np.shape}, {diff}'

            invert = inverts.at(j).item()
            windowing_param = windowing_params.at(j)
            voilut_func = voilut_funcs.at(j).item()
            voilut_func = VOILUT_FUNCS_INV_MAP[voilut_func]

            # YOLOX for ROI extraction
            img_yolox = min_max_scale(img_torch)
            img_yolox = (img_yolox * 255)  # float32
            if invert:
                img_yolox = 255 - img_yolox
            # YOLOX infer
            # who know if exception happen in hidden test ?
            try:
                xyxy, _area_pct, _conf = roi_extractor.detect_single(img_yolox)
                if xyxy is not None:
                    x0, y0, x1, y1 = xyxy
                    crop = img_torch[y0:y1, x0:x1]
                else:
                    crop = img_torch
            except:
                print('ROI extract exception!')
                crop = img_torch

            # apply windowing
            if windowing_param.shape[1] != 0:
                default_window_center = windowing_param[0, 0]
                default_window_width = windowing_param[1, 0]
                crop = apply_windowing(crop,
                                       window_width=default_window_width,
                                       window_center=default_window_center,
                                       voi_func=voilut_func,
                                       y_min=0,
                                       y_max=255,
                                       backend='torch')
            # if no window center/width in dcm file
            # do simple min-max scaling
            else:
                print('No windowing param!')
                crop = min_max_scale(crop)
                crop = crop * 255
            if invert:
                crop = 255 - crop
            crop = resize_and_pad(crop, MODEL_INPUT_SIZE)
            crop = crop.to(torch.uint8)
            crop = crop.cpu().numpy()
            save_img_to_file(save_path, crop, backend=save_backend)


#     assert cur_idx == len(
#         save_paths) - 1, f'{cur_idx} != {len(save_paths) - 1}'
    try:
        del external_source, pipe, roi_extractor
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return


def decode_and_save_dali_parallel(
        roi_yolox_engine_path,
        dcm_paths,
        save_paths,
        save_backend='cv2',
        batch_size=1,
        num_threads=1,
        py_num_workers=1,
        py_start_method='fork',
        device_id=0,
        parallel_n_jobs=2,
        parallel_n_chunks=4,
        parallel_backend='joblib',  # joblib or multiprocessing
        joblib_backend='loky'):
    assert parallel_backend in ['joblib', 'multiprocessing']
    assert joblib_backend in ['threading', 'multiprocessing', 'loky']
    # py_num_workers > 0 means using multiprocessing worker
    # 'fork' multiprocessing after CUDA init is not work (we must use 'spawn' instead)
    # since our pipeline can be re-build (when a dicom can't be decoded on GPU),
    # 2 options:
    #       (py_num_workers = 0, py_start_method=?)
    #       (py_num_workers > 0, py_start_method = 'spawn')
    assert not (py_num_workers > 0 and py_start_method == 'fork')

    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return decode_crop_save_dali(roi_yolox_engine_path,
                                     dcm_paths,
                                     save_paths,
                                     save_backend=save_backend,
                                     batch_size=batch_size,
                                     num_threads=num_threads,
                                     py_num_workers=py_num_workers,
                                     py_start_method=py_start_method,
                                     device_id=device_id)
    else:
        num_samples = len(dcm_paths)
        num_samples_per_chunk = num_samples // parallel_n_chunks
        if num_samples % parallel_n_chunks > 0:
            num_samples_per_chunk += 1
        starts = [num_samples_per_chunk * i for i in range(parallel_n_chunks)]
        ends = [
            min(start + num_samples_per_chunk, num_samples) for start in starts
        ]
        if isinstance(device_id, list):
            assert len(device_id) == parallel_n_chunks
        elif isinstance(device_id, int):
            device_id = [device_id] * parallel_n_chunks

        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`, {parallel_n_chunks} chunks ...'
        )
        if parallel_backend == 'joblib':
            _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
                delayed(decode_crop_save_dali)(
                    roi_yolox_engine_path,
                    dcm_paths[start:end],
                    save_paths[start:end],
                    save_backend=save_backend,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    py_num_workers=py_num_workers,  # ram_v3
                    py_start_method=py_start_method,
                    device_id=worker_device_id,
                ) for start, end, worker_device_id in zip(
                    starts, ends, device_id))
        else:  # manually start multiprocessing's processes
            workers = []
            daemon = False if py_num_workers > 0 else True
            for i in range(parallel_n_jobs):
                start = starts[i]
                end = ends[i]
                worker_device_id = device_id[i]
                worker = mp.Process(group=None,
                                    target=decode_crop_save_dali,
                                    args=(
                                        roi_yolox_engine_path,
                                        dcm_paths[start:end],
                                        save_paths[start:end],
                                    ),
                                    kwargs={
                                        'save_backend': save_backend,
                                        'batch_size': batch_size,
                                        'num_threads': num_threads,
                                        'py_num_workers': py_num_workers,
                                        'py_start_method': py_start_method,
                                        'device_id': worker_device_id,
                                    },
                                    daemon=daemon)
                workers.append(worker)
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
    return


def _single_decode_crop_save_sdl(roi_extractor,
                                 dcm_path,
                                 save_path,
                                 save_backend='cv2',
                                 index=0):
    dcm = dicomsdl.open(dcm_path)
    meta = DicomsdlMetadata(dcm)
    info = dcm.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]

    ori_dtype = info['dtype']
    img = np.empty(shape, dtype=ori_dtype)
    dcm.copyFrameData(index, img)
    img_torch = torch.from_numpy(img.astype(np.int16)).cuda()

    # YOLOX for ROI extraction
    img_yolox = min_max_scale(img_torch)
    img_yolox = (img_yolox * 255)  # float32
    # @TODO: subtract on large array --> should move after F.interpolate()
    if meta.invert:
        img_yolox = 255 - img_yolox
    # YOLOX infer
    try:
        xyxy, _area_pct, _conf = roi_extractor.detect_single(img_yolox)
        if xyxy is not None:
            x0, y0, x1, y1 = xyxy
            crop = img_torch[y0:y1, x0:x1]
        else:
            crop = img_torch
    except:
        print('ROI extract exception!')
        crop = img_torch

    # apply voi lut
    if meta.window_widths:
        crop = apply_windowing(crop,
                               window_width=meta.window_widths[0],
                               window_center=meta.window_centers[0],
                               voi_func=meta.voilut_func,
                               y_min=0,
                               y_max=255,
                               backend='torch')
    else:
        print('No windowing param!')
        crop = min_max_scale(crop)
        crop = crop * 255

    if meta.invert:
        crop = 255 - crop
    crop = resize_and_pad(crop, MODEL_INPUT_SIZE)
    crop = crop.to(torch.uint8)
    crop = crop.cpu().numpy()
    save_img_to_file(save_path, crop, backend=save_backend)


def decode_crop_save_sdl(roi_yolox_engine_path,
                         dcm_paths,
                         save_paths,
                         save_backend='cv2'):
    """DicomSDL decoding --> ROI cropping --> norm --> save as 8-bits PNG"""
    
    assert len(dcm_paths) == len(save_paths)
    roi_detector = roi_extract.RoiExtractor(engine_path=roi_yolox_engine_path,
                                            input_size=ROI_YOLOX_INPUT_SIZE,
                                            num_classes=1,
                                            conf_thres=ROI_YOLOX_CONF_THRES,
                                            nms_thres=ROI_YOLOX_NMS_THRES,
                                            class_agnostic=False,
                                            area_pct_thres=ROI_AREA_PCT_THRES,
                                            hw=ROI_YOLOX_HW,
                                            strides=ROI_YOLOX_STRIDES,
                                            exp=None)
    print('ROI extractor (YOLOX) loaded!')
    for i in tqdm(range(len(dcm_paths))):
        _single_decode_crop_save_sdl(roi_detector, dcm_paths[i], save_paths[i],
                                     save_backend)

    del roi_detector
    gc.collect()
    torch.cuda.empty_cache()
    return


def decode_crop_save_sdl_parallel(roi_yolox_engine_path,
                                  dcm_paths,
                                  save_paths,
                                  save_backend='cv2',
                                  parallel_n_jobs=2,
                                  parallel_n_chunks=4,
                                  joblib_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return decode_crop_save_sdl(roi_yolox_engine_path, dcm_paths,
                                    save_paths, save_backend)
    else:
        num_samples = len(dcm_paths)
        num_samples_per_chunk = num_samples // parallel_n_chunks
        if num_samples % parallel_n_chunks > 0:
            num_samples_per_chunk += 1
        starts = [num_samples_per_chunk * i for i in range(parallel_n_chunks)]
        ends = [
            min(start + num_samples_per_chunk, num_samples) for start in starts
        ]

        print(
            f'Starting {parallel_n_jobs} jobs with backend `{joblib_backend}`, {parallel_n_chunks} chunks...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
            delayed(decode_crop_save_sdl)(roi_yolox_engine_path,
                                          dcm_paths[start:end],
                                          save_paths[start:end], save_backend)
            for start, end in zip(starts, ends))
