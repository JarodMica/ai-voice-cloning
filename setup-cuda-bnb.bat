
git clone https://github.com/JarodMica/bitsandbytes-windows.git .\modules\bitsandbytes-windows\

xcopy .\modules\bitsandbytes-windows\bin\* .\venv\Lib\site-packages\bitsandbytes\. /Y
xcopy .\modules\bitsandbytes-windows\bin\cuda_setup\* .\venv\Lib\site-packages\bitsandbytes\cuda_setup\. /Y
xcopy .\modules\bitsandbytes-windows\bin\nn\* .\venv\Lib\site-packages\bitsandbytes\nn\. /Y

pause