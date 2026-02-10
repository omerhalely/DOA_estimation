@echo off
REM Test pretrained v1 model with FM encoding across different RT60 values
REM RT60 values: 0.2s to 1.3s (step: 0.1s)

echo ============================================================
echo Testing Pretrained V1 Model - FM Encoding
echo RT60 Test: 0.2s to 1.3s
echo ============================================================
echo.

REM Test 1: RT60 = 0.2s
echo [1/12] Running V1 FM - RT60 0.2s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.2.json --reverberation_time 0.2 --fine_tuned False
echo.

REM Test 2: RT60 = 0.3s
echo [2/12] Running V1 FM - RT60 0.3s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.3.json --reverberation_time 0.3 --fine_tuned False
echo.

REM Test 3: RT60 = 0.4s
echo [3/12] Running V1 FM - RT60 0.4s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.4.json --reverberation_time 0.4 --fine_tuned False
echo.

REM Test 4: RT60 = 0.5s
echo [4/12] Running V1 FM - RT60 0.5s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.5.json --reverberation_time 0.5 --fine_tuned False
echo.

REM Test 5: RT60 = 0.6s
echo [5/12] Running V1 FM - RT60 0.6s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.6.json --reverberation_time 0.6 --fine_tuned False
echo.

REM Test 6: RT60 = 0.7s
echo [6/12] Running V1 FM - RT60 0.7s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.7.json --reverberation_time 0.7 --fine_tuned False
echo.

REM Test 7: RT60 = 0.8s
echo [7/12] Running V1 FM - RT60 0.8s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.8.json --reverberation_time 0.8 --fine_tuned False
echo.

REM Test 8: RT60 = 0.9s
echo [8/12] Running V1 FM - RT60 0.9s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_0.9.json --reverberation_time 0.9 --fine_tuned False
echo.

REM Test 9: RT60 = 1.0s
echo [9/12] Running V1 FM - RT60 1.0s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_1.0.json --reverberation_time 1.0 --fine_tuned False
echo.

REM Test 10: RT60 = 1.1s
echo [10/12] Running V1 FM - RT60 1.1s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_1.1.json --reverberation_time 1.1 --fine_tuned False
echo.

REM Test 11: RT60 = 1.2s
echo [11/12] Running V1 FM - RT60 1.2s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_1.2.json --reverberation_time 1.2 --fine_tuned False
echo.

REM Test 12: RT60 = 1.3s
echo [12/12] Running V1 FM - RT60 1.3s...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\rt60\fm\results_v1_fm_rt60_1.3.json --reverberation_time 1.3 --fine_tuned False
echo.

echo ============================================================
echo All 12 RT60 tests completed for V1 FM!
echo ============================================================
echo.
echo Results saved to ./results/pretrained/rt60/fm/ directory:
echo   results_v1_fm_rt60_0.2.json ... results_v1_fm_rt60_1.3.json
echo.

pause
