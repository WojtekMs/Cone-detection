#!/usr/bin/env bash

#this script is supposed to perform all the necessary steps to create optimized TensorRT Engine from PyTorch weight file (.pts format)
#all operations will be kept inside build_engine directory
#this script will have to download 2 GitHub repositories, it will need sudo elevation level to serialize PyTorch based model to TensorRT .engine file
#if the user passes --remove flag, after all the steps are completed the script removes build_engine directory

manual() {
    
    echo "This script is supposed to perform all the necessary steps to create optimized TensorRT Engine from PyTorch weight file (.pts format)"
    echo -e "all operations will be kept inside build_engine directory\n"
    
    echo -e "script usage: $0 [options]\n"
    
    echo "Available options: "
    echo "-r (remove)"
    echo "it will remove build_engine directory after the script is finished"
    echo "build_engine directory may be around 200MB size, it is not necessary to use cone_detection"
    echo -e "however if you plan to compile another TensorRT engine soon, it will save you a lot of time if you do not remove build_engine dir\n"
    
    echo "-s (ssh)"
    echo "it will clone the repositories using SSH authorization"
    echo "this is the recommended way of cloning, however if you do not have SSH keys set up you do not have to use this"
    echo "by default cloning is done using login & password authorization"
}


REMOVE=''
SSH=''
WORKSPACE_ROOT=$(realpath .)
BUILD_ENGINE_WORKSPACE='build_engine'


#parse CLI arguments

#Reset in case getopts has been used previously in the shell.
OPTIND=1

while getopts "h?rs" opt; do
    case "$opt" in
    h|\?)
        manual
        exit 0
        ;;
    r)  REMOVE='true'
        ;;
    s)  SSH='true'
        ;;
    esac
done

shift $((OPTIND-1))


# prepare workspace
if [[ ! -e "$BUILD_ENGINE_WORKSPACE" ]]; then
    mkdir -p $BUILD_ENGINE_WORKSPACE
fi

# download necessary repositories
cd $BUILD_ENGINE_WORKSPACE
if [[ $SSH ]]; then
    git clone -b yolov5-v4.0-cone-detection git@github.com:WojtekMs/tensorrtx.git
    git clone -b v4.0-cone-detection git@github.com:WojtekMs/yolov5.git
else
    git clone -b yolov5-v4.0-cone-detection https://github.com/WojtekMs/tensorrtx.git
    git clone -b v4.0-cone-detection https://github.com/WojtekMs/yolov5.git 
fi

# copy prepared model weights and python script to yolov5 model repository
cp ${WORKSPACE_ROOT}/resources/cone_yolov5s.pt ${WORKSPACE_ROOT}/${BUILD_ENGINE_WORKSPACE}/yolov5 
cp ${WORKSPACE_ROOT}/${BUILD_ENGINE_WORKSPACE}/tensorrtx/yolov5/gen_wts.py ${WORKSPACE_ROOT}/${BUILD_ENGINE_WORKSPACE}/yolov5

# create .wts weights file from .pts weights file
cd ${WORKSPACE_ROOT}/${BUILD_ENGINE_WORKSPACE}/yolov5
python3 gen_wts.py 

# copy created .wts weights to tensorrtx repository
cp cone_yolov5s.wts ${WORKSPACE_ROOT}/${BUILD_ENGINE_WORKSPACE}/tensorrtx/yolov5

#build tensorrtx project
cd ${WORKSPACE_ROOT}/${BUILD_ENGINE_WORKSPACE}/tensorrtx/yolov5
mkdir build
cd build
cmake ..
make

#compile TensorRT engine from .wts file
./yolov5 -s ../cone_yolov5s.wts cone_yolov5s.engine s

#export TensorRT engine and plugins library to resources
mv cone_yolov5s.engine libmyplugins.so ${WORKSPACE_ROOT}/resources/

# cleanup after work
if [[ $REMOVE ]]; then
    rm -rf $BUILD_ENGINE_WORKSPACE
fi

echo "Script finished compiling, you should be ready to start cone_detection!"
