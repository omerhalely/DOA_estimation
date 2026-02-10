@echo off
REM Test pretrained v1 model with FM encoding across different SNR values
REM SNR values: 5, 10, 15, 20, 25, 30 dB

echo ============================================================
echo Testing Pretrained V1 Model - FM Encoding
echo SNR Test: 5-30 dB (with noise)
echo ============================================================
echo.

REM Test 1: SNR = 5 dB
echo [1/6] Running V1 FM - SNR 5 dB...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\snr\fm\results_v1_fm_snr_05.json --noise_probability 1.0 --snr_db 5 --fine_tuned False
echo.

REM Test 2: SNR = 10 dB
echo [2/6] Running V1 FM - SNR 10 dB...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\snr\fm\results_v1_fm_snr_10.json --noise_probability 1.0 --snr_db 10 --fine_tuned False
echo.

REM Test 3: SNR = 15 dB
echo [3/6] Running V1 FM - SNR 15 dB...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\snr\fm\results_v1_fm_snr_15.json --noise_probability 1.0 --snr_db 15 --fine_tuned False
echo.

REM Test 4: SNR = 20 dB
echo [4/6] Running V1 FM - SNR 20 dB...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\snr\fm\results_v1_fm_snr_20.json --noise_probability 1.0 --snr_db 20 --fine_tuned False
echo.

REM Test 5: SNR = 25 dB
echo [5/6] Running V1 FM - SNR 25 dB...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\snr\fm\results_v1_fm_snr_25.json --noise_probability 1.0 --snr_db 25 --fine_tuned False
echo.

REM Test 6: SNR = 30 dB
echo [6/6] Running V1 FM - SNR 30 dB...
python LibriSpeechInference.py --model_version v1 --mpe_type FM --output pretrained\snr\fm\results_v1_fm_snr_30.json --noise_probability 1.0 --snr_db 30 --fine_tuned False
echo.

echo ============================================================
echo All 6 SNR tests completed for V1 FM!
echo ============================================================
echo.
echo Results saved to ./results/pretrained/snr/fm/ directory:
echo   results_v1_fm_snr_05.json ... results_v1_fm_snr_30.json
echo.

pause
