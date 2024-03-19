portable_git\bin\git.exe clone https://github.com/JarodMica/ai-voice-cloning.git
cd ai-voice-cloning
git submodule init
git submodule update --remote
cd ..

xcopy ai-voice-cloning\update_package.bat /E /I /H /Y

xcopy ai-voice-cloning\src src /E /I /H /Y
xcopy ai-voice-cloning\modules\dlas modules\dlas /E /I /H /Y
xcopy ai-voice-cloning\modules\tortoise-tts modules\tortoise-tts /E /I /H /Y

@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo Finished updating!
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pause