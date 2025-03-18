#!/bin/bash

echo Today is $(date) and is a beautiful day
echo Powered by Luca Zelioli

conda activate maati_transformer

# export var for python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTHONPATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

#path_list=../Configuration/Masterfiles/Inputs_test_area.csv

echo Insert area
read -r area

echo insert the path of the inputs
read -r path_list

echo Override windoiws dimension YES or NO
read -r override

python create_dataset.py -area "${area}" -path_list "${path_list}" -override_channel "${override}"
