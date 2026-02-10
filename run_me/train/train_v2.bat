@echo off
REM Train fine-tuned v1 model with specific configuration
REM Max microphones: 8
REM Noise probability: 0.2
REM Epochs: 300-350

python FineTune.py --model_name v2 --model_version v2 --noise_probability 1.0 --start_epoch 0 --epochs 300

echo.
echo ============================================================
echo Training completed!
echo ============================================================
echo.

pause
