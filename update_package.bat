portable_git\bin\git.exe clone https://github.com/JarodMica/ai-voice-cloning.git
xcopy ai-voice-cloning\update_package.bat /E /I /H /Y

xcopy ai-voice-cloning\src src /E /I /H /Y

rmdir /s /q ai-voice-cloning
rmdir /s /q .git

pause