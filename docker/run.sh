#!/bin/bash

xhost +local:root; \
    nvidia-docker run -it --rm --shm-size=16G \
    -e DISPLAY=$DISPLAY \
    -e CUDACXX=/opt/cuda/bin/nvcc \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $PWD/../:/workspace/src:rw \
    -v $PWD/../../dataset:/workspace/dataset:rw \
    -v /dev/bus/usb:/dev/bus/usb \
    telef-alignment


