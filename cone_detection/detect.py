import pycuda.driver as cuda
import time
import tensorrt as trt
import numpy as np

from cone_detection.preprocess import preprocess_img
from cone_detection.postprocess import postprocess
from cone_detection.utils import transform_detected_coords_to_original

def detect(engine: trt.ICudaEngine, img: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    #this function performs network execution on the given img
    #additionally this function does preprocessing and postprocessing of img
    #param engine: tensor rt engine created from network weights
    #param img: image to perform detection on
    #return value: predictions in original image coordinates
    #predictions: bounding boxes, confidences, class_ids

    with engine.create_execution_context() as context:
        h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        #preprocess
        preprocess_start = time.time()
        preprocessed_img, resize_ratio, padding_size = preprocess_img(img)
        preprocess_stop = time.time()
        #copy our input image to buffer
        np.copyto(h_input, preprocessed_img.flatten())
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        postprocess_start = time.time()
        predictions = postprocess(h_output)
        predictions_in_original_coords = transform_detected_coords_to_original(predictions, resize_ratio, padding_size)
        postprocess_stop = time.time()
        print(f"Preprocessing time: {(preprocess_stop - preprocess_start) * 1000:.4f} ms")
        print(f"Postprocessing time: {(postprocess_stop - postprocess_start) * 1000:.4f} ms")
        print(f"Complete detection time: {(postprocess_stop - preprocess_start) * 1000:.4f} ms")
    return predictions_in_original_coords
