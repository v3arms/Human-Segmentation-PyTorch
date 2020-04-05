call C:\\%homepath%\miniconda3\Scripts\activate.bat

call conda create -n humanseg python=3.6
call conda activate humanseg
call pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
call pip install -r requirements.txt

call pause