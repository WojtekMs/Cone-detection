#additional debug info will be printed when True
DEBUG = False

#this value affects the minimum size of input image (it must be divisable by stride)
NETWORK_STRIDE = 32

BLACK_COLOR = (114, 114, 114)
BLUE_COLOR = (255, 0, 0)
ORANGE_COLOR = (0, 140, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)

#when images width > height we only do padding on top & bottom of an image
LEFT_RIGHT_PADDING = 0

#the size of an image after preprocessing (height, width)
DESIRED_IMG_SIZE = (416, 416)

#path from workspace root to tensor rt network engine
engine_file_path = 'resources/cone_yolov5s.engine'

#path from workspace root to necessary plugins for tensor rt
#necessary to use yolov5 model with tensor rt
plugins_lib_path = 'resources/libmyplugins.so'

#path from workspace root to output folder
#if it doesnt exist it will be automatically created
#it will be automatically overwritten when app is started
SAVE_PATH = "data/output/"

#this value is used by NMS algorithm
#if area of intersection is greater than 0.5 of the best bounding box
#then remove this bounding box (as it is just noise)
IOU_THRESHOLD = 0.5

#network returns only ID of detected classes, here we give them names
class_names = ['cone']
