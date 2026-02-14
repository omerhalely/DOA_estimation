@echo off
REM Train v1 model with specific configuration
REM Noise probability: 1.0
REM Epochs: 0-300

REM Training with PM (Positional Mapping) encoding
python FineTune.py --model_name v1 --model_version v1 --mpe_type PM --noise_probability 1.0 --epochs 300

REM Training with FM (Feature Mapping) encoding
python FineTune.py --model_name v1 --model_version v1 --mpe_type FM --noise_probability 1.0 --epochs 300

echo.
echo ============================================================
echo Training completed!
echo ============================================================
echo.
