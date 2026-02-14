@echo off
REM Train v2 model with specific configuration
REM Noise probability: 1.0
REM Epochs: 0-350

REM Training v2 model
python FineTune.py --model_name v2 --model_version v2 --noise_probability 1.0 --start_epoch 0 --epochs 350

echo.
echo ============================================================
echo Training completed!
echo ============================================================
echo.
