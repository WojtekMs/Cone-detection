#!/usr/bin/env python3

import cv2
import numpy as np
import time
import torch

DEBUG = False

def resize_keep_aspect_ratio_YOLOv5VERSION(img, new_size: tuple):
    # this function does the same padding as in original detect.py
    # this function assumes that incoming pictures are rectangular (width > height)
    # h,w = im.shape
    # height = new_size[0]
    r = min(new_size[0] / img.shape[0], new_size[1] / img.shape[1])
    new_size_with_ratio = (round(img.shape[0] * r), round(img.shape[1] * r))
    resized = cv2.resize(img, (new_size_with_ratio[1], new_size_with_ratio[0]))
    top_bottom_padding = ((new_size[0] - resized.shape[0]) % 32) // 2
    return cv2.copyMakeBorder(resized, top_bottom_padding, top_bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(114,114,114))

def resize_keep_aspect_ratio(img, new_size: tuple):
    # this function does padding so that img has 416x416 exact size
    # this function assumes that incoming pictures are rectangular (width > height)
    # h,w = im.shape
    # height = new_size[0]
    r = min(new_size[0] / img.shape[0], new_size[1] / img.shape[1])
    new_size_with_ratio = (round(img.shape[0] * r), round(img.shape[1] * r))
    resized = cv2.resize(img, (new_size_with_ratio[1], new_size_with_ratio[0]))
    top_bottom_padding = (new_size[0] - resized.shape[0]) // 2
    return cv2.copyMakeBorder(resized, top_bottom_padding, top_bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=(114,114,114))


def preprocess_img(img):
    resized = resize_keep_aspect_ratio(img, (416, 416))
    rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
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
    return batch_channel_height_width_tensor

if __name__ == "__main__":
    desired_size = (416, 416)
    img_name = "photo.jpg"
    img = cv2.imread(img_name)

    preprocessed = preprocess_img(img)
