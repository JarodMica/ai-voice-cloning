@echo off

set ffmpeg_url=https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z?download=true
set ffmpeg_folder=ffmpeg
set ffmpeg_zip=ffmpeg.7z

curl -o 7z.exe  "https://www.7-zip.org/a/7zr.exe"

if not exist "%ffmpeg_folder%" (
        if not exist "%ffmpeg_zip%" (
            echo Downloading %ffmpeg_zip%...
            curl -L -o "%ffmpeg_zip%" "%ffmpeg_url%"
            if errorlevel 1 (
                echo Download failed. Please check your internet connection or the URL and try again.
                exit /b 1
            )
        ) else (
            echo File %ffmpeg_zip% already exists, skipping download.
        )
        
        echo Extracting %ffmpeg_zip%...
        7z.exe x %ffmpeg_zip% -o%ffmpeg_folder%
        echo FFmpeg has finished downloading and extracting.
    ) else (
        echo FFmpeg folder %ffmpeg_folder% already exists, skipping download and extraction.
    )

:: Move ffmpeg.exe and ffprobe.exe to the ffmpeg folder root
for /D %%i in ("%ffmpeg_folder%\*") do (
    if exist "%%i\bin\ffmpeg.exe" move "%%i\bin\ffmpeg.exe" "ffmpeg.exe"
    if exist "%%i\bin\ffprobe.exe" move "%%i\bin\ffprobe.exe" "ffprobe.exe"
)

echo FFmpeg moved out of downloaded folder