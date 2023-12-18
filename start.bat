call .\venv\Scripts\activate.bat
set PATH=.\bin\;%PATH%
set PYTHONUTF8=1
python .\src\main.py %*
pause