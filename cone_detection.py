#!/usr/bin/env python3

import os
import sys
import time

from numpy.lib.npyio import save
import tensorrt as trt
import ctypes
import pycuda.driver as cuda
import numpy as np
import torch
import cv2

from cone_detection.preprocess import preprocess_img
from cone_detection.postprocess import postprocess
from cone_detection.detect import detect
from cone_detection.utils import draw_detections, transform_detected_coords_to_original
from cone_detection import config

def manual():
    print(f"program usage: {sys.argv[0]} images_directory")


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
engine_file_path = config.engine_file_path
plugins_lib_path = config.plugins_lib_path
ctypes.CDLL(os.path.abspath(plugins_lib_path))
SAVE_PATH = config.SAVE_PATH

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        manual()
        exit(0)
    if os.path.isdir(sys.argv[1]):
        path = sys.argv[1]
        img_paths = [os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    else:
        img_paths = [sys.argv[1]]
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    with trt.Runtime(TRT_LOGGER) as runtime:
        with open(engine_file_path, 'rb') as engine_file:
            print(f'Deserializing engine {engine_file_path}, please wait..')
            engine = runtime.deserialize_cuda_engine(engine_file.read())
            print(f'Engine deserialized!')
            print("-"*30)
            for img_path in img_paths:
                print(f'Image: {img_path}')
                img = cv2.imread(img_path)
                detections, resize_ratio, padding_size = detect(engine, img)
                print(f'Number of detections: {len(detections[0])}')
                img = draw_detections(img, transform_detected_coords_to_original(detections, resize_ratio, padding_size))
                img_save_path = os.path.join(SAVE_PATH, img_path.split('/')[-1])
                cv2.imwrite(img_save_path, img)
                print(f"Img is saved to {img_save_path}")
                print("-"*30)
    ctx.pop()
