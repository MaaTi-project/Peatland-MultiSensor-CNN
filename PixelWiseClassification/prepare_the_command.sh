#!/bin/bash

echo Today is $(date) and is a beautiful day
echo Powered by Luca Zelioli

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTHONPATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

conda activate maati_transformer

echo Write the area you want to analyse
read -r area

echo Write pixel size in meter
read -r override_pix

echo Write how many cpu / cores you have
read -r override_cpu

echo Write path to the dataset Optinal parameters Press enter to skip
read -r path_to_dataset

python pixelwise_classification_create_commands.py -area "${area}" --opix "${override_pix}" --ocpu "${override_cpu}" --path_to_raster "${path_to_dataset}"


