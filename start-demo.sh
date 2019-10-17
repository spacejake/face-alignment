#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

eval "$(conda shell.bash hook)"
conda activate jake

pushd $DIR/examples

args=$@
if [ -z "$@" ]; then
  args="--max-faces 5"
fi

python detect_cam.py $args

popd

