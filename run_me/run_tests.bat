@echo off
REM Master test runner - executes all test batch files
REM This script runs all inference tests for the GI-DOAEnet models

echo ============================================================
echo GI-DOAEnet - Master Test Runner
echo ============================================================
echo.
echo This will run all tests in the following order:
echo   1. V1 PM SNR Tests (5-30 dB)
echo   2. V1 PM RT60 Tests (0.2s-1.3s)
echo   3. V1 FM SNR Tests (5-30 dB)
echo   4. V1 FM RT60 Tests (0.2s-1.3s)
echo.
echo ============================================================
echo.

REM Run all test batch files in sequence
echo Starting Test Suite...
echo.

REM Test 1: V1 PM SNR
echo [1/4] Running V1 PM SNR Tests...
call run_me\tests\test_v1_pm_snr.bat
echo.

REM Test 2: V1 PM RT60
echo [2/4] Running V1 PM RT60 Tests...
call run_me\tests\test_v1_pm_rt60.bat
echo.

REM Test 3: V1 FM SNR
echo [3/4] Running V1 FM SNR Tests...
call run_me\tests\test_v1_fm_snr.bat
echo.

REM Test 4: V1 FM RT60
echo [4/4] Running V1 FM RT60 Tests...
call run_me\tests\test_v1_fm_rt60.bat
echo.

echo ============================================================
echo All Test Suites Completed!
echo ============================================================
echo.
echo Test results are saved in the ./results/ directory:
echo   ./results/pretrained/snr/pm/
echo   ./results/pretrained/snr/fm/
echo   ./results/pretrained/rt60/pm/
echo   ./results/pretrained/rt60/fm/
echo.
