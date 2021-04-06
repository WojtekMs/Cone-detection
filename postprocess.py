#!/usr/env/bin python3

import torch
import torchvision

IOU_THRESHOLD = 0.5

class_names = ['cone']

def postprocess(output):
    #output is flat ndarray which is output from network
    #number of detections, center x, center y, width, height, conf, class id, ...
    #returns list of predictions for network output
    #boxes (xyxy), confidences, class_ids
    tensor_output = torch.from_numpy(output)
    detection_per_row = tensor_output[1:].reshape(-1, 6)[:int(tensor_output[0])]
    print(detection_per_row.shape)
    xyxy_boxes = torchvision.ops.box_convert(detection_per_row[:, :4], 'cxcywh', 'xyxy')
    indices = torchvision.ops.nms(xyxy_boxes, detection_per_row[:, 4], IOU_THRESHOLD)
    best_boxes = xyxy_boxes[indices]
    confs = detection_per_row[indices, 4]
    class_ids = detection_per_row[indices, 5].int()
    for id, box in enumerate(best_boxes):
        print(f'box no {id} top left: {best_boxes[id][:2]}, bottom right: {best_boxes[id][2:]}, conf: {confs[id]}, class: {class_names[class_ids[id]]}')
    return best_boxes, confs, class_ids
