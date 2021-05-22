# Cone detection
Currently the program performs cone detection from given input images.  
It is using serialized platform specific TensorRT engine created from YOLOv5 PyTorch based network weights (.pt).  
The application does image preprocessing, inference and  results postprocessing.    
Furthermore it converts returned bounding boxes coordinates to original image size and draws them on the original image.  
Lastly, marked images are saved into the output path defined in config.py

# Dependencies
Since this application uses TensorRT engine, you have to compile the engine from network weights, so that the engine is compliant with your platform. In order to do so you need:
- C++17 Compiler
- CMake >= 3.13
- CUDA >= 10.0
- TensorRT >= 7.0
- OpenCV >= 4.1.2

Additionally you need to download all the other requirements specified in the requirements.txt using:
- `pip3 install -r requirements.txt`

# Building
This application uses CUDA library, hence it is necessary to have Nvidia Graphics Card in order to build & run this project.  
1. setup virtual environment
- `python3 -m virtualenv venv`
- `source venv/bin/activate`

2. install dependencies
- `pip install -r requirements.txt`

3. compile your TensorRT engine
- `./compile_engine.sh -r`

4. launch the application
- `./cone_detection.py data/input`

# Examples
![9](https://github.com/WojtekMs/Cone-detection/blob/master/data/output/9.jpg?raw=true)
![4](https://github.com/WojtekMs/Cone-detection/blob/master/data/output/4.jpg?raw=true)

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
