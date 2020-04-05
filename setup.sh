eval "$(conda shell.bash hook)"

conda create -n humanseg python=3.6
conda activate humanseg
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
