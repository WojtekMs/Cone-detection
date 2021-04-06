#!/usr/bin/env python3

import os
import sys
import time
import tensorrt as trt
import ctypes
import pycuda.driver as cuda
import numpy as np
import torch
import cv2

from preprocess import preprocess_img
from postprocess import postprocess
from detect import detect


def manual():
    print(f"program usage: {sys.argv[0]} images_directory")


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
engine_file_path = 'cone_yolov5s.engine'
plugins_lib_path = 'libmyplugins.so'
ctypes.CDLL(os.path.abspath(plugins_lib_path))


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        manual()
        exit(0)
    if os.path.isdir(sys.argv[1]):
        path = sys.argv[1]
        img_names = [os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    else:
        img_names = [sys.argv[1]]
    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    with trt.Runtime(TRT_LOGGER) as runtime:
        with open(engine_file_path, 'rb') as engine_file:
            print(f'Deserializing engine {engine_file_path}, please wait..')
            engine = runtime.deserialize_cuda_engine(engine_file.read())
            print(f'Engine deserialized!')
            with engine.create_execution_context() as context:
                for img_name in img_names:
                    img = cv2.imread(img_name)
                    try:
                        detect(context, img)
                    except:
                        ctx.pop()
    ctx.pop()
