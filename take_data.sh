#!/bin/bash

config=$1
fname=$2
save_dir=$3

mkdir -p $save_dir

python3 py-scope.py $config $fname $save_dir
