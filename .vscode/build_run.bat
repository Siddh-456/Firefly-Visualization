@echo off
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
cd /d "%~1"

REM Kill any running instance so the exe isn't locked
taskkill /F /IM "%~3.exe" >nul 2>&1
taskkill /F /IM "nq_new.exe" >nul 2>&1
timeout /T 1 /NOBREAK >nul

if exist "JobShop.h" (
    echo Compiling JobShopGA project (SFML)...
    g++ -std=c++17 *.cpp -lsfml-graphics -lsfml-window -lsfml-system -o JobShopGA.exe
    if %ERRORLEVEL% EQU 0 (
        echo Running JobShopGA...
        JobShopGA.exe
    )
) else (
    echo Compiling %~2 (Raylib)...
    g++ -std=c++17 "%~2" -lraylib -lopengl32 -lgdi32 -lwinmm -o "%~3.exe"
    if %ERRORLEVEL% EQU 0 (
        echo Running %~3...
        start "" "%~3.exe"
    ) else (
        echo BUILD FAILED — check errors above
    )
)
