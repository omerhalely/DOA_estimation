@echo off
REM Master training runner - executes all training batch files
REM This script runs all trainings for the GI-DOAEnet models

echo ============================================================
echo GI-DOAEnet - Master Training Runner
echo ============================================================
echo.
echo This will run all trainings in the following order:
echo   1. V1
echo   2. V2
echo.
echo ============================================================
echo.

REM Run all training batch files in sequence
echo Starting Training Suite...
echo.

REM Training 1: V1
echo [1/2] Running V1 Training...
call run_me\train\train_v1.bat
echo.

REM Training 2: V2
echo [2/2] Running V2 Training...
call run_me\train\train_v2.bat
echo.

echo ============================================================
echo All Training Completed!
echo ============================================================
echo.
