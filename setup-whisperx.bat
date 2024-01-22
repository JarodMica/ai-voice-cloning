@echo off
call .\venv\Scripts\activate.bat

:: Install whisperx
python -m pip install git+https://github.com/m-bain/whisperx.git


:: Fix incompatibilites caused from packages in whisperx
python -m pip install -r requirements.txt

echo Installation finished
pause