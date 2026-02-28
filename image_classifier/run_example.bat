@echo off
REM Run AI vs Human detector on an image.
REM Usage: run_example.bat [image_path]
REM Default image: ..\example.jpeg
setlocal
set IMG=%~1
if "%IMG%"=="" set IMG=..\example.jpeg
cd /d "%~dp0"

if exist "venv\Scripts\python.exe" (
  call venv\Scripts\python.exe predict.py "%IMG%"
) else (
  echo Creating venv and installing deps (one-time)...
  python -m venv venv
  call venv\Scripts\pip.exe install -q torch>=2.1.0 transformers Pillow accelerate safetensors
  call venv\Scripts\python.exe predict.py "%IMG%"
)
endlocal
