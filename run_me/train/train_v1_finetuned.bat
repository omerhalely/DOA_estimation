@echo off
REM Train fine-tuned v1 model with specific configuration
REM Max microphones: 8
REM Noise probability: 0.2
REM Epochs: 300-350

python FineTune.py --model_name "v1_finetuned" --model_version v1 --mpe_type PM --noise_probability 1.0 --start_epoch 300 --epochs 350 --resume --checkpoint_name "v1"

echo.
echo ============================================================
echo Training completed!
echo ============================================================
echo.

pause
