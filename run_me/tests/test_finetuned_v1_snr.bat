@echo off
REM Test fine-tuned v1 model across different SNR values
REM SNR values: 5, 10, 15, 20, 25, 30 dB

echo ============================================================
echo Testing Fine-Tuned V1 Model - PM Encoding
echo SNR Test: 5-30 dB (with noise)
echo ============================================================
echo.

@REM REM Test 1: SNR = 5 dB
@REM echo [1/6] Running Fine-Tuned V1 PM - SNR 5 dB...
@REM python LibriSpeechInference.py --model_version v1 --mpe_type PM --output finetuned\snr\results_finetuned_v1_pm_snr_05.json --noise_probability 1.0 --snr_db 5 --fine_tuned True
@REM echo.

@REM REM Test 2: SNR = 10 dB
@REM echo [2/6] Running Fine-Tuned V1 PM - SNR 10 dB...
@REM python LibriSpeechInference.py --model_version v1 --mpe_type PM --output finetuned\snr\results_finetuned_v1_pm_snr_10.json --noise_probability 1.0 --snr_db 10 --fine_tuned True
@REM echo.

@REM REM Test 3: SNR = 15 dB
@REM echo [3/6] Running Fine-Tuned V1 PM - SNR 15 dB...
@REM python LibriSpeechInference.py --model_version v1 --mpe_type PM --output finetuned\snr\results_finetuned_v1_pm_snr_15.json --noise_probability 1.0 --snr_db 15 --fine_tuned True
@REM echo.

@REM REM Test 4: SNR = 20 dB
@REM echo [4/6] Running Fine-Tuned V1 PM - SNR 20 dB...
@REM python LibriSpeechInference.py --model_version v1 --mpe_type PM --output finetuned\snr\results_finetuned_v1_pm_snr_20.json --noise_probability 1.0 --snr_db 20 --fine_tuned True
@REM echo.

@REM REM Test 5: SNR = 25 dB
@REM echo [5/6] Running Fine-Tuned V1 PM - SNR 25 dB...
@REM python LibriSpeechInference.py --model_version v1 --mpe_type PM --output finetuned\snr\results_finetuned_v1_pm_snr_25.json --noise_probability 1.0 --snr_db 25 --fine_tuned True
@REM echo.

REM Test 6: SNR = 30 dB
echo [6/6] Running Fine-Tuned V1 PM - SNR 30 dB...
python LibriSpeechInference.py --model_name v1 --model_version v1 --mpe_type PM --output finetuned\snr\results_finetuned_v1_pm_snr_30.json --noise_probability 1.0 --snr_db 30 --fine_tuned True
echo.

echo ============================================================
echo All 6 SNR tests completed for Fine-Tuned V1 PM!
echo ============================================================
echo.
echo Results saved to ./results/finetuned/snr/ directory:
echo   results_finetuned_v1_pm_snr_05.json ... results_finetuned_v1_pm_snr_30.json
echo.

pause
