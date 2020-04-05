conda create -n humanseg python=3.6
conda activate humanseg
conda install pytorch==1.2.0 cudatoolkit==9.2
conda install Pillow
pip install ffmpeg-python
pip install -r requirements.txt -U
