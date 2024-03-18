portable_git\bin\git.exe clone https://github.com/JarodMica/ai-voice-cloning.git
cd ai-voice-cloning
git submodule init
git submodule update --remote
cd ..

xcopy ai-voice-cloning\update_package.bat /E /I /H /Y

xcopy ai-voice-cloning\src src /E /I /H /Y

rmdir /s /q ai-voice-cloning
rmdir /s /q .git

@echo Finished updating!
pause