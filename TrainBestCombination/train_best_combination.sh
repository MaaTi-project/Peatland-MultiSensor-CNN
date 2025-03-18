echo Today is $(date) and is a beautiful day
echo Powered by Luca Zelioli

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export PYTHONPATH="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

conda activate maati_transformer

echo Write the area you want to analyse
read -r area

echo Do you want to me to run the drained or undrained?
read -r type


python train_best_combination.py -area "${area}" -types "${type}" 