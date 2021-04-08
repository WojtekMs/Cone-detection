#!/usr/bin/env python3

import cv2
import sys
import numpy as np
import time
import torch
from cone_detection import config

DEBUG = config.DEBUG
NETWORK_STRIDE = config.NETWORK_STRIDE
BLACK_COLOR = config.BLACK_COLOR
LEFT_RIGHT_PADDING = config.LEFT_RIGHT_PADDING
DESIRED_IMG_SIZE = config.DESIRED_IMG_SIZE

def resize_keep_aspect_ratio_YOLOv5VERSION(img, new_size: tuple):
    # this function does the same padding as in original YOLOv5 detect.py
    # in original YOLOv5 detect.py when new_size = (416,416) output img size = (256, 416)
    # this function assumes that incoming pictures are rectangular (width > height)
    # h,w = im.shape
    # height = new_size[0]

    ratio = min(new_size[0] / img.shape[0], new_size[1] / img.shape[1])
    new_size_with_ratio = (round(img.shape[0] * ratio),
                           round(img.shape[1] * ratio))
    resized = cv2.resize(img, (new_size_with_ratio[1], new_size_with_ratio[0]))
    top_bottom_padding = (
        (new_size[0] - resized.shape[0]) % NETWORK_STRIDE) // 2
    return (cv2.copyMakeBorder(resized,
                              top_bottom_padding,
                              top_bottom_padding,
                              LEFT_RIGHT_PADDING,
                              LEFT_RIGHT_PADDING,
                              cv2.BORDER_CONSTANT,
                              value=BLACK_COLOR), ratio, top_bottom_padding)


def resize_keep_aspect_ratio(img, new_size: tuple):
    # this function does padding so that img has exactly new_size
    # this function assumes that incoming pictures are rectangular (width > height)
    # h,w = im.shape
    # height = new_size[0]
    # --------------------------
    # params:
    #img: the original image you want to resize (ndarray),
    #new_size: the size you want to achieve after resize (tuple)
    # return value:
    #resized and padded image (ndarray), resize ratioi (float), padding size (int)
    ratio = min(new_size[0] / img.shape[0], new_size[1] / img.shape[1])
    new_size_with_ratio = (round(img.shape[0] * ratio),
                           round(img.shape[1] * ratio))
    resized = cv2.resize(img, (new_size_with_ratio[1], new_size_with_ratio[0]))
    top_bottom_padding = (new_size[0] - resized.shape[0]) // 2
    assert (
        (resized.shape[0] + top_bottom_padding * 2) % NETWORK_STRIDE == 0
    ), f"Padding has been calculated badly! New image size must be divisible by {NETWORK_STRIDE}!\
         Whereas new image size is: {resized.shape[0] + top_bottom_padding * 2}"
    return (cv2.copyMakeBorder(resized,
                              top_bottom_padding,
                              top_bottom_padding,
                              LEFT_RIGHT_PADDING,
                              LEFT_RIGHT_PADDING,
                              cv2.BORDER_CONSTANT,
                              value=BLACK_COLOR), ratio, top_bottom_padding)


def preprocess_img(img):
    #this function prepares our input data to be processed by network
    # - resize and pad 
    # - convert color to RGB
    # - transform img from (height, width, channel) to (channel, height, width) notation
    # - scale the pixel values down to 0 - 1
    # - transform img to (batch_size, channel, height, width) notation
    #--------------------------------------------------------------
    #param img: image to be preprocessed (ndarray from cv2.imread)
    #return value: resized and padded img (tensor), resize ratio (float), padding size (int)
    resized_img, ratio, padding = resize_keep_aspect_ratio(img, DESIRED_IMG_SIZE)
    rgb_resized = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    channel_height_width_img = rgb_resized.transpose(2, 0, 1)
    normalized = channel_height_width_img / 255.0
    tensor_normalized = torch.from_numpy(normalized).float()
    batch_channel_height_width_tensor = tensor_normalized.unsqueeze(0)
    if DEBUG:
        print(f'preprocessed img')
        print(f'shape: {batch_channel_height_width_tensor.shape}')
        print(f'type: {batch_channel_height_width_tensor.dtype}')
        print(f'device: {batch_channel_height_width_tensor.device}')
        print(f'layout: {batch_channel_height_width_tensor.layout}')
    return (batch_channel_height_width_tensor, ratio, padding)

def manual():
    print(f"program usage: {sys.args[0]} img")

if __name__ == "__main__":
    desired_size = (416, 416)
    if len(sys.args) < 2 or len(sys.args) > 2:
        manual()
        exit(0)
    img_name = sys.args[1]
    img = cv2.imread(img_name)
    preprocessed = preprocess_img(img, desired_size)
