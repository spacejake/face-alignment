#!/bin/bash

python train.py \
    --pointType 3D \
    --checkpoint ./checkpoint2 \
    --resume ./checkpoint2/test_checkpointFAN.pth.tar \
    --resume-depth ./checkpoint2/test_checkpointDepth.pth.tar \
    --devices 0,1,2
