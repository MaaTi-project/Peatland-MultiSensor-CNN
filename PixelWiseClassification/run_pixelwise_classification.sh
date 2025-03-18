#!/bin/bash

echo Today is $(date) and is a beautiful day
echo Powered by Luca Zelioli

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTHONPATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

conda activate maati_transformer


echo "Enter the path of the launch commander"
read -r file_path

# Check if the file exists
if [[ ! -f "$file_path" ]]; then
  echo "[EXCEPTION] Error: File does not exist."
  exit 1
fi

# Execute commands one by one
while IFS= read -r cmd; do
  echo "Executing: $cmd"
  $cmd
done < "${file_path}"
