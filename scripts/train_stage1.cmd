@echo off
setlocal

set "VSLANG=1033"
chcp 65001 >nul
set "PYTHONUTF8=1"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64
if errorlevel 1 exit /b %errorlevel%
set "VSLANG=1033"

for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "PYTHONPATH=%PROJECT_ROOT%\src"
set "TRITON_CACHE_DIR=%PROJECT_ROOT%\.cache\triton"
set "TORCHINDUCTOR_CACHE_DIR=%PROJECT_ROOT%\.cache\torchinductor"

cd /d "%PROJECT_ROOT%"
"E:\anaconda3\envs\MC_Gen\python.exe" -u scripts\train.py --model configs\model\base.yaml --train configs\train\stage_1.yaml %*
