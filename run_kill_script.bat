cd /d %~dp0
call venv\Scripts\activate
python kill_pid.py
pause