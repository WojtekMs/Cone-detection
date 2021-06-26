import tensorrt as trt
import ctypes
import os
import pycuda.driver as cuda
import numpy as np

from cone_detection import config
from cone_detection.detect import detect

#this class manages a resource - CUDA context, therefore it should be used with Python resource manager
# with YoloDetector(engine, plugin) as detector:
#   ...

class YoloDetector(object):
    def __init__(self, engine_file_path: str, plugins_lib_path: str):
        cuda.init()
        ctypes.CDLL(os.path.abspath(plugins_lib_path))
        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.engine_file_path = engine_file_path
        self.plugins_lib_path = plugins_lib_path
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        with trt.Runtime(self.TRT_LOGGER) as runtime:
            with open(self.engine_file_path, 'rb') as engine_file:
                print(f'Deserializing engine {self.engine_file_path}, please wait..')
                self.engine = runtime.deserialize_cuda_engine(engine_file.read())
                print(f'Engine deserialized!')
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.ctx.pop()
    
    def detect(self, img: np.ndarray):
        return detect(self.engine, img)
