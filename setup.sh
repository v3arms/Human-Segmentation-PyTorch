eval "$(conda shell.bash hook)"

conda create -n humanseg python=3.6
conda activate humanseg
pip install -r requirements.txt
