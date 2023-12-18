call .\venv\Scripts\activate.bat
set PYTHONUTF8=1
python ./src/train.py --yaml "%1"
pause
deactivate