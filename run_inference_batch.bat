@echo off
REM Batch file to run GI-DOAEnet inference with different microphone counts
REM Tests both PM and FM encodings for 4-12 microphones

echo ============================================================
echo Running GI-DOAEnet Inference - Microphone Count Tests
echo ============================================================
echo.

REM PM Encoding Tests (4-12 microphones)
echo ========== PM Encoding Tests ==========
echo.

REM Test 1: PM - 4 mics
echo [1/18] Running V1 PM - 4 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_4mics.json --num_microphones 4
echo.

REM Test 2: PM - 5 mics
echo [2/18] Running V1 PM - 5 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_5mics.json --num_microphones 5
echo.

REM Test 3: PM - 6 mics
echo [3/18] Running V1 PM - 6 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_6mics.json --num_microphones 6
echo.

REM Test 4: PM - 7 mics
echo [4/18] Running V1 PM - 7 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_7mics.json --num_microphones 7
echo.

REM Test 5: PM - 8 mics
echo [5/18] Running V1 PM - 8 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_8mics.json --num_microphones 8
echo.

REM Test 6: PM - 9 mics
echo [6/18] Running V1 PM - 9 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_9mics.json --num_microphones 9
echo.

REM Test 7: PM - 10 mics
echo [7/18] Running V1 PM - 10 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_10mics.json --num_microphones 10
echo.

REM Test 8: PM - 11 mics
echo [8/18] Running V1 PM - 11 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_11mics.json --num_microphones 11
echo.

REM Test 9: PM - 12 mics
echo [9/18] Running V1 PM - 12 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type PM --output results_v1_pm_12mics.json --num_microphones 12
echo.

REM FM Encoding Tests (4-12 microphones)
echo ========== FM Encoding Tests ==========
echo.

REM Test 10: FM - 4 mics
echo [10/18] Running V1 FM - 4 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_4mics.json --num_microphones 4
echo.

REM Test 11: FM - 5 mics
echo [11/18] Running V1 FM - 5 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_5mics.json --num_microphones 5
echo.

REM Test 12: FM - 6 mics
echo [12/18] Running V1 FM - 6 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_6mics.json --num_microphones 6
echo.

REM Test 13: FM - 7 mics
echo [13/18] Running V1 FM - 7 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_7mics.json --num_microphones 7
echo.

REM Test 14: FM - 8 mics
echo [14/18] Running V1 FM - 8 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_8mics.json --num_microphones 8
echo.

REM Test 15: FM - 9 mics
echo [15/18] Running V1 FM - 9 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_9mics.json --num_microphones 9
echo.

REM Test 16: FM - 10 mics
echo [16/18] Running V1 FM - 10 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_10mics.json --num_microphones 10
echo.

REM Test 17: FM - 11 mics
echo [17/18] Running V1 FM - 11 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_11mics.json --num_microphones 11
echo.

REM Test 18: FM - 12 mics
echo [18/18] Running V1 FM - 12 microphones...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output results_v1_fm_12mics.json --num_microphones 12
echo.

echo ============================================================
echo All 18 inference runs completed!
echo ============================================================
echo.
echo Results saved to ./results/ directory:
echo   PM: results_v1_pm_4mics.json ... results_v1_pm_12mics.json
echo   FM: results_v1_fm_4mics.json ... results_v1_fm_12mics.json
echo.

pause
