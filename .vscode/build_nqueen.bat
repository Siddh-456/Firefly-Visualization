@echo off
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
cd /d "%~dp0.."

taskkill /F /IM nqueen.exe >nul 2>&1

echo [Compiling nqueen.cpp...]
g++ -std=c++17 -O0 nqueen.cpp -lraylib -lopengl32 -lgdi32 -lwinmm -o nqueen.exe

if %ERRORLEVEL% EQU 0 (
    echo [BUILD OK -- Launching]
    start "" nqueen.exe
) else (
    echo [BUILD FAILED]
)
