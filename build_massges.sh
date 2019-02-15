#!/bin/bash

declare msg_file=./messages/messages.proto
declare gen_dir=./messages/gen

rm $gen_dir -rf
mkdir -p $gen_dir

# Python
declare py_dir=$gen_dir/python
mkdir $py_dir -p
protoc -I=. --python_out=$py_dir $msg_file



