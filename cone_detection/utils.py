import cv2
from cone_detection import config

def get_color(confidence):
    color = ()
    if confidence >= 0.80:
        color = config.GREEN_COLOR
    if confidence >= 0.5 and confidence < 0.80:
        color = config.ORANGE_COLOR
    if confidence < 0.5:
        color = config.RED_COLOR
    return tuple(color)

def transform_detected_coords_to_original(detections, resize_ratio, padding_size):
    #this function converts coordinates from resized img executed by network
    #back to original img coordinates
    #----------------------------------------------------    
    #params:
    #detections: this is output from detect function (tuple)
    #resize_ratio: ratio that was used to resize image
    #padding_size: how big was the padding (value of top or bottom padding)
    #----------------------------------------------------
    #return value: converted coordinates, resize_ratio, padding_size
    xyxy = detections[0]
    xyxy[:, 0] /= resize_ratio
    xyxy[:, 2] /= resize_ratio
    xyxy[:, 1] = (xyxy[:, 1] - padding_size) / resize_ratio
    xyxy[:, 3] = (xyxy[:, 3] - padding_size) / resize_ratio
    return xyxy, detections[1], detections[2]

def draw_detections(img, detections, line_width = 3):
    xyxy, conf, class_id = detections
    for xyxy, conf, class_id in zip(xyxy, conf, class_id):
        img = cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), get_color(conf), line_width)
        img = cv2.putText(img, 
                          f"{config.class_names[class_id]} {conf: .2f}", 
                          org=(xyxy[0], xyxy[1] - 5), 
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale=0.5, 
                          color=config.BLUE_COLOR, 
                          thickness=2)
    return img

