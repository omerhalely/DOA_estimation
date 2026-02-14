# Geometry-Invariant DOA Estimation Network (GI-DOAEnet)

This repository contains implementation and pretrained weights for **"DNN-based Geometry-Invariant DOA Estimation with Microphone Positional Encoding and Complexity Gradual Training"** [[1]](#references).

<img src="./figures/architecture.jpg" alt="Overall architecture" width="600"/>

Overall architecture of Geometry-Invariant DOA Estimation Network (GI-DOAEnet) with Microphone Positional Encoding (MPE). With $C$-channel signals and the coordinates of microphones, the geometry-invariant network structure estimates the Direction of Arrival (DOA).

---

## Table of Contents
- [Features](#features)
- [Microphone Positional Encoding (MPE)](#microphone-positional-encoding-mpe)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Model Versions](#model-versions)
- [Examples](#examples)
- [References](#references)

---

## Features

- **Geometry-Invariant Architecture**: Works with arbitrary microphone array configurations (4-12 microphones)
- **Two Positional Encoding Types**: Phase Modulation (PM) and Frequency Modulation (FM)
- **Multiple Model Versions**:
  - **V1**: Azimuth-only DOA estimation
  - **V2**: Joint azimuth and elevation DOA estimation
- **Curriculum Learning**: 3-phase gradual training with increasing complexity
- **Flexible Testing**: Comprehensive test suites for SNR and RT60 robustness analysis
- **GPU-Accelerated**: Full PyTorch implementation with GPU support via gpuRIR

---

## Microphone Positional Encoding (MPE)

<img src="./figures/v_eq.png" alt="v_eq">

### Phase Modulation (PM)
<img src="./figures/PM_eq.png" alt="PM_eq">

### Frequency Modulation (FM)
<img src="./figures/FM_eq.png" alt="FM_eq">

Where:
- $M$ is the latent feature size
- $r_{c}$, $\theta_{c}$, and $\phi_{c}$ are the distance, azimuth, and elevation angles of the $c$-th microphone
- $\alpha$ is an amplitude scaling factor
- $\beta$ is a frequency scaling factor

---

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- PyTorch 1.8+

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd GI-DOAEnet
```

2. **Install PyTorch**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Install dependencies**:
```bash
pip install numpy scipy tqdm matplotlib tensorboard
```

4. **Build gpuRIR** (for GPU-accelerated room impulse response simulation):
```bash
cd gpuRIR
pip install .
cd ..
```

---

## Dataset Preparation

This project uses the **LibriSpeech** dataset for training and evaluation.

1. **Download LibriSpeech**:
   - Training: `train-clean-100` or `train-clean-360`
   - Validation: `dev-clean`
   - Testing: `test-clean`

2. **Build the database**:
```bash
python build_database.py
```

3. **Data structure**:
```
data/
├── train/
├── validation/
└── test/
```

The dataset automatically generates:
- Random room dimensions (3x3x2.5m to 10x8x6m)
- Random microphone array geometries
- Random source positions
- Room impulse responses (RT60: 0.2-1.3s)
- Additive noise with configurable SNR

---

## Quick Start

### Basic Inference

Run inference on the test set with pretrained V1 model (PM encoding):
```bash
python LibriSpeechInference.py --model_version v1 --mpe_type PM
```

### Simple Training Example

Train V1 model with PM encoding:
```bash
python FineTune.py --model_name v1 --model_version v1 --mpe_type PM --epochs 300
```

---

## Training

### Using Batch Scripts (Recommended)

#### Train All Models
```bash
run_me/run_train.bat
```

This runs:
1. **V1 Training** (both PM and FM encodings)
2. **V2 Training** (azimuth + elevation)

#### Individual Training Scripts

**Train V1 models**:
```bash
run_me/train/train_v1.bat
```

**Train V2 model**:
```bash
run_me/train/train_v2.bat
```

### Training Parameters

Key command-line arguments for `FineTune.py`:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Model name for saving | `v1` |
| `--model_version` | Model version (`v1` or `v2`) | `v1` |
| `--mpe_type` | Positional encoding (`PM` or `FM`) | `PM` |
| `--epochs` | Total training epochs | `300` |
| `--start_epoch` | Starting epoch (for resuming) | `0` |
| `--batch_size` | Training batch size | `16` |
| `--lr` | Initial learning rate | `2.5e-4` |
| `--noise_probability` | Probability of adding noise | `1.0` |
| `--max_microphones` | Maximum number of microphones | `12` |

### Curriculum Learning Strategy

Training follows a 3-phase curriculum:

| Phase | Epochs | Microphones | Geometry | LR | Weight Decay |
|-------|--------|-------------|----------|-----|--------------|
| **1** | 1-5 | Fixed (4) | Fixed | 2.5e-4 | 1e-4 |
| **2** | 6-10 | Fixed (4) | Random | 5e-4 | 1e-6 |
| **3** | 11+ | Random (4-12) | Random | 1e-3 | 1e-6 |

---

## Testing

### Run All Tests
```bash
run_me/run_tests.bat
```

This executes all test suites:
1. V1 PM - SNR Tests (5-30 dB)
2. V1 PM - RT60 Tests (0.2-1.3s)
3. V1 FM - SNR Tests (5-30 dB)
4. V1 FM - RT60 Tests (0.2-1.3s)

### Individual Test Scripts

**SNR Robustness Tests** (tests across different noise levels):
```bash
run_me/tests/test_v1_pm_snr.bat   # V1 PM encoding, SNR: 5-30 dB
run_me/tests/test_v1_fm_snr.bat   # V1 FM encoding, SNR: 5-30 dB
```

**RT60 Robustness Tests** (tests across different reverberation times):
```bash
run_me/tests/test_v1_pm_rt60.bat  # V1 PM encoding, RT60: 0.2-1.3s
run_me/tests/test_v1_fm_rt60.bat  # V1 FM encoding, RT60: 0.2-1.3s
```

### Test Results

Results are saved as JSON files in:
```
results/
├── pretrained/
│   ├── snr/
│   │   ├── pm/    # PM encoding SNR test results
│   │   └── fm/    # FM encoding SNR test results
│   └── rt60/
│       ├── pm/    # PM encoding RT60 test results
│       └── fm/    # FM encoding RT60 test results
└── finetuned/     # Fine-tuned model test results
```

### Inference Parameters

Key command-line arguments for `LibriSpeechInference.py`:

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_version` | Model version (`v1` or `v2`) | `v1` |
| `--mpe_type` | Positional encoding (`PM` or `FM`) | `PM` |
| `--fine_tuned` | Use fine-tuned model | `False` |
| `--output` | Output JSON file path | `results.json` |
| `--noise_probability` | Probability of adding noise | `1.0` |
| `--snr_db` | Signal-to-noise ratio in dB | `30` |
| `--reverberation_time` | Fixed RT60 in seconds | `None` (random) |
| `--num_microphones` | Fixed number of microphones | `None` (random) |

---

## Project Structure

```
GI-DOAEnet/
├── model/                          # Model architecture
│   ├── main.py                     # GI_DOAEnet main model
│   ├── FFT.py                      # STFT processing
│   ├── Microphone_positional_encoding.py
│   ├── Channel_invariant_feature_extractor.py
│   ├── SpatioTemporal_block.py     # For V2 (azimuth + elevation)
│   ├── SpatioTemporal_block_v1.py  # For V1 (azimuth only)
│   ├── Spatial_spectrum_mapping.py
│   └── RoPE.py                     # Rotary positional encoding
│
├── run_me/                         # Batch execution scripts
│   ├── train/                      # Training scripts
│   │   ├── train_v1.bat           # Train V1 (PM + FM)
│   │   └── train_v2.bat           # Train V2
│   ├── tests/                      # Test scripts
│   │   ├── test_v1_pm_snr.bat
│   │   ├── test_v1_pm_rt60.bat
│   │   ├── test_v1_fm_snr.bat
│   │   └── test_v1_fm_rt60.bat
│   ├── run_train.bat              # Execute all training
│   └── run_tests.bat              # Execute all tests
│
├── FineTune.py                     # Training script
├── LibriSpeechInference.py         # Inference script
├── LibriSpeechDataset.py           # Dataset with RIR simulation
├── util.py                         # Utility functions
├── build_database.py               # Dataset preparation
├── inference.py                    # Simple inference demo
│
├── pretrained/                     # Pretrained model weights
│   ├── GI_DOAEnet_PM.tar
│   └── GI_DOAEnet_FM.tar
│
├── saved_models/                   # Trained model checkpoints
├── results/                        # Test results (JSON)
├── runs/                           # TensorBoard logs
├── data/                           # LibriSpeech dataset
├── gpuRIR/                         # GPU Room Impulse Response library
└── figures/                        # Architecture diagrams
```

---

## Model Versions

### V1 Model (Azimuth-only)
- **Output**: 360-degree azimuth estimation
- **Use case**: Standard DOA estimation in 2D plane
- **Pretrained weights**: Available for both PM and FM encodings
- **Performance**: 
  - PM: ~4.1 degrees MAE
  - FM: ~4.5 degrees MAE

### V2 Model (Azimuth + Elevation)
- **Output**: 360-degree azimuth + elevation estimation
- **Use case**: Full 3D DOA estimation
- **Requires**: Training from scratch (no pretrained weights)

---

## Examples

### Training Example
```bash
# Train V1 model with PM encoding for 300 epochs
python FineTune.py \
  --model_name my_v1_pm \
  --model_version v1 \
  --mpe_type PM \
  --epochs 300 \
  --batch_size 16 \
  --noise_probability 1.0
```

### Inference Example
```bash
# Test V1 PM model with SNR=20dB
python LibriSpeechInference.py \
  --model_version v1 \
  --mpe_type PM \
  --noise_probability 1.0 \
  --snr_db 20 \
  --output results/my_test.json
```

### Spectrum Visualization
<table>
  <tr>
    <td align="center">
      <img src="./spectrum_plots/FM/10ch_0.png" alt="FM 10ch 0" width="500"/><br/>
      FM example with 10 channels and 1 speaker.
    </td>
    <td align="center">
      <img src="./spectrum_plots/PM/4ch_1.png" alt="PM 4ch 1" width="500"/><br/>
      PM example with 4 channels and 2 speakers.
    </td>
  </tr>
</table>

---

## References

<a name="references"></a>

[1] M.-S. Baek, J.-H. Chang and I. Cohen, "DNN-based Geometry-Invariant DOA Estimation with Microphone Positional Encoding and Complexity Gradual Training," *IEEE Trans. Audio, Speech, Lang. Process.*, vol. 33, pp. 2360-2376, 2025, doi: [10.1109/TASLPRO.2025.3577336](https://doi.org/10.1109/TASLPRO.2025.3577336).

---

## License

Please refer to the original paper for citation and usage terms.
