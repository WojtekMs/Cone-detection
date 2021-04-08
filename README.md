# Cone detection
Currently the program performs cone detection from given input images.  
It is using serialized tensor rt engine created from yolov5 network weights.  
It does preprocessing, inference and postprocessing.  
Furthermore it converts returned bounding boxes coordinates to original image size and draws them on the original image.  
Lastly, marked images are saved into the output path defined in config.py

## Notes
- memory allocated by pycuda.driver does not need to be freed explicitly (object frees memory when is being destroyed)
- tensor rt context is like creating a new UNIX process (can be used for multi-threading) at least one context is necessary to run application
- during training YOLOv5 model input images are not exactly 416,416 size!  
perhaps it is worthwhile to edit the code and change that to achieve better results

## TODO
- add feature that will allow to detect cones from video stream instead from single images
- think about batch size (currently it is working at batch size = 1)
- think about Deep Learning Accelerator (can it be used to optimize inference?)
- think about multi-threading
- think about weight precision (currently float32? can be float16 or int8)  
(engine was supposed to work on float16 but it is not certain what precision is really used in the program)
