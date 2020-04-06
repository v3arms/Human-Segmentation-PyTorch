call C:\\%homepath%\miniconda3\Scripts\activate.bat

call conda env remove -n humanseg
call rmdir /s C:\\%homepath%\miniconda3\envs\humanseg

call pause
