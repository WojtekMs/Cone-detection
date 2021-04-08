#!/usr/env/bin python3

import torch
import torchvision
from cone_detection import config


def postprocess(output, resize_ratio, padding_size):
    #this function transforms the raw values returned by network inference 
    #into usable values containing bounding boxes, confidences and class ids
    #----------------------------------------------------------
    #param: output: it is flat ndarray which is output from network
    #output format: 
    #[number of detections, 
    #center x, center y, width, height, conf, class id,
    #center x, center y, width, height, conf, class id, ...]
    #-----------------------------------------------------------
    #return value: list of predictions for network output
    #boxes (xyxy), confidences, class_ids
    #return value format:
    #([
    #[x1, y1, x2, y2],
    #[x1, y1, x2, y2], ...], [conf, conf, ...], [class_id, class_id, ...])
    
    tensor_output = torch.from_numpy(output)
    detection_per_row = tensor_output[1:].reshape(-1, 6)[:int(tensor_output[0])]
    xyxy_boxes = torchvision.ops.box_convert(detection_per_row[:, :4], 'cxcywh', 'xyxy')
    indices = torchvision.ops.nms(xyxy_boxes, detection_per_row[:, 4], config.IOU_THRESHOLD)
    best_boxes = xyxy_boxes[indices]
    confs = detection_per_row[indices, 4]
    class_ids = detection_per_row[indices, 5].int()
    return (best_boxes, confs, class_ids)
