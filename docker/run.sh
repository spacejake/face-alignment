#!/bin/bash

declare data_path=$1

echo "Data Path: $data_path"

xhost +local:root; \
    nvidia-docker run -it --rm --shm-size=16G \
    -e DISPLAY=$DISPLAY \
    -e CUDACXX=/opt/cuda/bin/nvcc \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $PWD/../:/workspace/src:rw \
    -v $data_path:/workspace/dataset:rw \
    -v /dev/bus/usb:/dev/bus/usb \
    --ipc host \
    --env QT_X11_NO_MITSHM=1 \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/bus/usb/003/015:/dev/ttyUSB0 \
    morto067/pytorch-cuda9.0-ubuntu16.04


