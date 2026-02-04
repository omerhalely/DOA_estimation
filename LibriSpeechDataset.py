import os
import numpy as np
import math
import soundfile as sf
from gpuRIR import gpuRIR as grir
from scipy.signal import fftconvolve
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
import webrtcvad


class LibriSpeechDataset(Dataset):
    def __init__(self, data_path, mode, device, max_microphones=12, sample_rate=16000, training_phase=1, logger=None, noise_probability=0.0, snr_db=20.0):
        """
        LibriSpeech dataset with room impulse response simulation.
        
        Args:
            data_path: Base path to the data directory
            mode: One of 'train', 'validation', or 'test'
            sample_rate: Audio sample rate in Hz (default: 16000)
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
            training_phase: Training curriculum phase (1, 2, or 3):
                Phase 1: Fixed 4 microphones with fixed geometry
                Phase 2: Fixed 4 microphones with random geometry
                Phase 3: Random microphones (4-max) with random geometry
            logger: Optional logger for training messages
            noise_probability: Probability of adding Gaussian noise to each microphone (0.0 to 1.0)
            snr_db: Target Signal-to-Noise Ratio in dB when noise is added (default: 20.0)
        """
        self.data_path = data_path
        self.mode = mode
        self.sample_rate = sample_rate
        self.target_duration = 4.0  # 4 seconds
        self.target_samples = int(self.sample_rate * self.target_duration)
        self.sound_velocity = 343.0  # Speed of sound in m/s
        self.device = device
        self.max_microphones = max_microphones
        self.logger = logger  # Optional logger for training messages
        
        # Noise configuration
        self.noise_probability = noise_probability
        self.snr_db = snr_db

        self.num_microphones = 4
        
        # Training phase management
        self.training_phase = training_phase
        self._validate_phase()
        
        # Fixed microphone coordinates for Phase 1 (generated once)
        self._fixed_mic_coords = None
        if self.training_phase == 1:
            self._initialize_fixed_mics()

        self.vad = webrtcvad.Vad(3)
        self.frame_duration = 30  # ms

        assert mode in ["train", "validation", "test"], \
            f"Mode must be one of 'train', 'validation', or 'test', got {mode}"

        # Map mode to directory name
        if mode == "train":
            folder_name = "combined-train-clean-100"
        elif mode == "validation":
            folder_name = "combined-dev-clean"
        elif mode == "test":
            folder_name = "combined-test-clean"

        self.data_path = os.path.join(self.data_path, folder_name)
        self.files = [f for f in os.listdir(self.data_path) if f.endswith('.flac')]
    
    def _validate_phase(self):
        """Validate that the training phase is valid."""
        if self.training_phase not in [1, 2, 3]:
            raise ValueError(f"Training phase must be 1, 2, or 3, got {self.training_phase}")
    
    def set_training_phase(self, phase):
        """
        Change the training phase.
        
        Args:
            phase: New training phase (1, 2, or 3)
        """
        self.training_phase = phase
        self._validate_phase()
        
        # Initialize fixed mics for Phase 1 if switching to it
        if phase == 1 and self._fixed_mic_coords is None:
            self._initialize_fixed_mics()
        
        
        message = f"Training phase set to {phase}"
        if phase == 1:
            message += " -> Fixed 4 microphones with fixed geometry"
        elif phase == 2:
            message += " -> Fixed 4 microphones with random geometry"
        elif phase == 3:
            message += f" -> Random microphones (4-{self.max_microphones}) with random geometry"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _initialize_fixed_mics(self):
        """
        Generate a fixed microphone array configuration for Phase 1.
        Uses the same generation logic but stores the result.
        """
        num_channels = 4
        c_min, c_max = 4, 12
        progress = (num_channels - c_min) / (c_max - c_min)
        r_min = max(1, 4 - 3 * progress) / 100.0
        r_max = max(7, 9 + 4 * progress) / 100.0
        # r_min = torch.empty(1, device=self.device).uniform_(r_min, 6).item() / 100.0
        # r_max = torch.empty(1, device=self.device).uniform_(7, r_max).item() / 100.0
        
        mpos_local = []
        while len(mpos_local) < num_channels:
            candidate = torch.empty(3, device=self.device).uniform_(-2 * r_max, 2 * r_max)
            if len(mpos_local) == 0:
                mpos_local.append(candidate)
            else:
                dists = [torch.norm(candidate - m).item() for m in mpos_local]
                if all(r_min <= d <= r_max for d in dists):
                    mpos_local.append(candidate)
        
        mpos_local = torch.stack(mpos_local)
        mpos_local -= mpos_local.mean(dim=0)  # Center array
        # r_jitter = torch.empty(mpos_local.shape, device=self.device).uniform_(-0.5 / 100, 0.5 / 100)
        # mpos_local += r_jitter
        
        # Random rotation
        rotation_matrix = torch.from_numpy(R.random().as_matrix()).float().to(self.device)
        mpos_local = (rotation_matrix @ mpos_local.T).T
        
        self._fixed_mic_coords = mpos_local
        print(f"Initialized fixed microphone array for Phase 1 with {num_channels} microphones")

    def __len__(self):
        return len(self.files)
    
    def _normalize_audio_length(self, audio):
        if len(audio) < self.target_samples:
            # Pad with zeros
            padding = self.target_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        elif len(audio) > self.target_samples:
            # Truncate
            audio = audio[:self.target_samples]
        
        return audio
    
    def _frame_generator(self, audio, frame_duration_ms):
        """
        Generate audio frames for VAD processing.
        
        Args:
            audio: Numpy array of int16 audio samples
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
        
        Yields:
            Bytes of audio frames
        """
        n = int(self.sample_rate * (frame_duration_ms / 1000.0))
        offset = 0
        while offset + n <= len(audio):
            yield audio[offset:offset + n].tobytes()
            offset += n
    
    def sample_num_microphones(self):
        """Sample number of microphones based on current training phase."""
        if self.training_phase in [1, 2]:
            # Phase 1 & 2: Always use 4 microphones
            self.num_microphones = 4
        else:
            # Phase 3: Random number of microphones
            self.num_microphones = torch.randint(4, self.max_microphones + 1, (1,), device=self.device).item()
    
    def set_num_microphones(self, num_microphones):
        self.num_microphones = num_microphones
    
    def _add_gaussian_noise(self, audio_channel, vad_channel):
        """
        Add Gaussian noise to an audio channel with specified SNR.
        
        Args:
            audio_channel: Single channel audio tensor (T,) on self.device
            vad_channel: Voice activity detection tensor (T,) on self.device
            
        Returns:
            Noisy audio channel with target SNR
        """
        # Calculate signal power during speech regions only (for proper SNR scaling)
        speech_indices = vad_channel == 1
        
        if speech_indices.sum() == 0:
            # No speech detected - skip noise addition or use full signal power
            signal_power = torch.mean(audio_channel ** 2)
        else:
            signal_speech = audio_channel[speech_indices]
            signal_power = torch.mean(signal_speech ** 2)
        
        # Avoid division by zero
        if signal_power < 1e-10:
            # Signal is too quiet, return original audio
            return audio_channel
        
        # Calculate target noise power from SNR
        # SNR_dB = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        
        # Generate Gaussian noise for ENTIRE signal (not just speech)
        noise = torch.randn_like(audio_channel)
        
        # Scale noise to achieve target SNR
        current_noise_power = torch.mean(noise ** 2)
        noise = noise * torch.sqrt(noise_power / current_noise_power)
        
        # Add noise to ENTIRE signal (both speech and silence)
        noisy_audio = audio_channel + noise
        
        return noisy_audio

    def _generate_random_room(self):
        # 1. Randomize Room Dimensions (Table III)
        room_sz = torch.tensor([
            torch.empty(1, device=self.device).uniform_(3.0, 10.0).item(),  # Width
            torch.empty(1, device=self.device).uniform_(3.0, 8.0).item(),   # Length
            torch.empty(1, device=self.device).uniform_(2.5, 6.0).item()    # Height
        ], dtype=torch.float32, device=self.device)
        
        # 2. Randomize RT60 (Table III)
        rt60 = torch.empty(1, device=self.device).uniform_(0.2, 1.3).item()
        fs = 16000  # Sampling rate used in paper

        # 3. Dynamic Microphone Geometry (Section III-A)
        num_channels = self.num_microphones
        
        # Phase-dependent microphone generation
        if self.training_phase == 1:
            # Phase 1: Use fixed microphone coordinates
            mpos_local = self._fixed_mic_coords.clone()
        else:
            # Phase 2 & 3: Generate random microphone geometry
            # Equation (25) for R_min and R_max
            c_min, c_max = 4, 12
            progress = (num_channels - c_min) / (c_max - c_min)
            r_min = max(1, 4 - 3 * progress) / 100.0
            r_max = max(7, 9 + 4 * progress) / 100.0
            # r_min = torch.empty(1, device=self.device).uniform_(r_min, 6).item() / 100.0
            # r_max = torch.empty(1, device=self.device).uniform_(7, r_max).item() / 100.0

            mpos_local = []
            while len(mpos_local) < num_channels:
                candidate = torch.empty(3, device=self.device).uniform_(-r_max, r_max)
                if len(mpos_local) == 0:
                    mpos_local.append(candidate)
                else:
                    dists = [torch.norm(candidate - m).item() for m in mpos_local]
                    if all(r_min <= d <= r_max for d in dists):
                        mpos_local.append(candidate)
                    # else:
                    #     print("Failed to generate valid microphone geometry")
            
            mpos_local = torch.stack(mpos_local)  # (num_channels, 3) on self.device
            mpos_local -= mpos_local.mean(dim=0)  # Center array
            # r_jitter = torch.empty(mpos_local.shape, device=self.device).uniform_(-0.5 / 100, 0.5 / 100)
            # mpos_local += r_jitter
            
            # Random rotation using scipy (keep this as is, rotation matrix is small)
            rotation_matrix = torch.from_numpy(R.random().as_matrix()).float().to(self.device)
            mpos_local = (rotation_matrix @ mpos_local.T).T  # Random rotation
        
        # 4. Source Position (Table III)
        margin = 0.1
        dist_max = (min(room_sz[0].item(), room_sz[1].item(), room_sz[2].item()) / 2) - margin
        dist = torch.empty(1, device=self.device).uniform_(0.3, dist_max).item()
        theta = torch.empty(1, device=self.device).uniform_(0, 360).item()  # Azimuth
        phi = torch.empty(1, device=self.device).uniform_(30, 150).item()   # Elevation
        
        theta_rad, phi_rad = torch.deg2rad(torch.tensor(theta, device=self.device)), torch.deg2rad(torch.tensor(phi, device=self.device))
        spos_local = torch.tensor([
            dist * torch.sin(phi_rad) * torch.cos(theta_rad),
            dist * torch.sin(phi_rad) * torch.sin(theta_rad),
            dist * torch.cos(phi_rad)
        ], dtype=torch.float32, device=self.device)

        # 5. Room Placement with 0.1m Buffer
        # Calculate the maximum extent needed to fit both the array AND the source
        array_radius = torch.norm(mpos_local, dim=1).max().item()
        source_distance = torch.norm(spos_local).item()  # Distance from array center to source
        max_extent = max(array_radius, source_distance)  # Furthest point from anchor
        safe_margin = max_extent + margin  # Add 0.1m buffer
        
        anchor_min_x = min(safe_margin, room_sz[0].item() - safe_margin)
        anchor_max_x = max(safe_margin, room_sz[0].item() - safe_margin)
        anchor_min_y = min(safe_margin, room_sz[1].item() - safe_margin)
        anchor_max_y = max(safe_margin, room_sz[1].item() - safe_margin)
        anchor_min_z = min(safe_margin, room_sz[2].item() - safe_margin)
        anchor_max_z = max(safe_margin, room_sz[2].item() - safe_margin)
        anchor = torch.tensor([
            torch.empty(1, device=self.device).uniform_(anchor_min_x, anchor_max_x).item(),
            torch.empty(1, device=self.device).uniform_(anchor_min_y, anchor_max_y).item(),
            torch.empty(1, device=self.device).uniform_(anchor_min_z, anchor_max_z).item()
        ], dtype=torch.float32, device=self.device)

        mpos_global = mpos_local + anchor
        spos_global = (spos_local + anchor).unsqueeze(0)  # gpuRIR expects (S, 3)

        # 6. gpuRIR Simulation Parameters 
        # Trev: Image source method up to 12 dB attenuation 
        # Tmax: Diffused reverberation model up to 40 dB
        trev = 0.2 * rt60  # Approximate 12dB point
        tmax = 0.67 * rt60  # Approximate 40dB point
        
        # Convert to numpy for gpuRIR (it expects numpy arrays)
        room_sz_np = room_sz.clone().cpu().numpy()
        mpos_global_np = mpos_global.clone().cpu().numpy()
        spos_global_np = spos_global.clone().cpu().numpy()
        
        nb_img = grir.t2n(trev, room_sz_np)  # Reflection order for ISM part
        beta = grir.beta_SabineEstimation(room_sz_np, rt60)

        # 7. Generate RIRs
        rirs = grir.simulateRIR(
            room_sz=room_sz_np,
            beta=beta,
            pos_src=spos_global_np,
            pos_rcv=mpos_global_np,
            nb_img=nb_img,
            Tmax=tmax,
            fs=fs,
            Tdiff=trev
        )
        d = {
            "source": torch.tensor([dist, theta_rad, phi_rad], dtype=torch.float32, device=self.device),
            "receiver": {
                "mic_coords": mpos_local,
                "num_mics": num_channels
            }
        }
        return rirs[0], d

    def __getitem__(self, idx):
        """
        Load a FLAC file, simulate room acoustics, and return multi-channel audio.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with:
                - 'audio': Multi-channel audio tensor on self.device (num_mics, target_samples)
                - 'mic_coords': Microphone coordinates tensor on self.device (num_mics, 3)
                - 'source_coords': Source position tensor on self.device (1, 3)
                - 'num_mics': Number of microphones (int)
        """
        # Load FLAC file
        file_path = os.path.join(self.data_path, self.files[idx])
        audio, sr = sf.read(file_path)
        
        # Ensure correct sample rate
        if sr != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sr}")
        
        # Normalize to 4 seconds
        audio = self._normalize_audio_length(audio)

        # Convert to int16 for VAD (webrtcvad requires 16-bit PCM)
        # Normalize to [-1, 1] first, then scale to int16 range
        audio_normalized = audio / (np.abs(audio).max() + 1e-8)  # Prevent division by zero
        audio_int16 = (audio_normalized * 32767).astype(np.int16)
        
        # Extract VAD using webrtcvad
        vad_list = []
        for frame_idx, frame in enumerate(self._frame_generator(audio_int16, self.frame_duration)):
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            vad_list.append(1.0 if is_speech else 0.0)
            if frame_idx <= 2:
                vad_list[frame_idx] = 0.0
                
        # Resample VAD to match audio length (64000 samples)
        vad_tensor = torch.tensor(vad_list, dtype=torch.float32)
        if len(vad_tensor) > 0:
            indices = torch.linspace(0, len(vad_tensor) - 1, audio.shape[-1])
            vad = vad_tensor[torch.round(indices).long()]
        else:
            # Fallback: all speech
            vad = torch.ones(audio.shape[-1], dtype=torch.float32)
        vad = vad.unsqueeze(0).expand(self.num_microphones, vad.shape[-1])
        
        # Use original float audio for RIR simulation
        # Generate random room and get RIR (returns numpy from gpuRIR)
        rir_filters, positions = self._generate_random_room()
        num_mics = positions["receiver"]["num_mics"]
        
        # GPU-accelerated convolution using PyTorch
        # Convert to PyTorch tensors and move to GPU
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, samples)
        rir_tensor = torch.from_numpy(rir_filters).float().unsqueeze(1).to(self.device)  # (num_mics, 1, rir_length)
        
        # Perform batch convolution on GPU (all mics at once!)
        conv_result = F.conv1d(
            audio_tensor.expand(1, num_mics, -1),  # (1, num_mics, samples)
            rir_tensor.flip(-1),  # Flip RIR for convolution (conv vs correlation)
            padding=rir_tensor.shape[-1] - 1,  # 'full' mode equivalent
            groups=num_mics
        )  # Output: (1, num_mics, conv_length)
        
        # Extract and truncate to target length
        multi_channel_audio = conv_result[0, :, :self.target_samples]
        
        # Pad if any channel is shorter than target (edge case)
        if multi_channel_audio.shape[1] < self.target_samples:
            padding = self.target_samples - multi_channel_audio.shape[1]
            multi_channel_audio = F.pad(multi_channel_audio, (0, padding), mode='constant', value=0)
        
        # Add Gaussian noise if configured
        if torch.rand(1).item() < self.noise_probability:
            for mic_idx in range(num_mics):
                multi_channel_audio[mic_idx] = self._add_gaussian_noise(
                    multi_channel_audio[mic_idx],
                    vad[mic_idx]
                )
        
        params = {
            'mic_coords': positions["receiver"]["mic_coords"],
            'source_coords': positions["source"],
            'num_mics': num_mics
        }
        return multi_channel_audio, vad, params



if __name__ == "__main__":
    print("Testing LibriSpeechDataset...")
    
    # Set data path (adjust this to your actual data path)
    data_path = os.path.join(os.getcwd(), "data")
    mode = "validation"
    max_microphones = 12
    sample_rate = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with noise enabled
    print("\n" + "="*60)
    print("Testing with Gaussian Noise")
    print("="*60)
    noise_probability = 1.0  # 100% probability for testing
    snr_db = 30.0
    
    # Create dataset instance
    dataset = LibriSpeechDataset(
        data_path=data_path,
        mode=mode,
        sample_rate=sample_rate,
        device=device,
        noise_probability=noise_probability,
        snr_db=snr_db
    )

    dataset.set_training_phase(3)
    dataset.sample_num_microphones()
    
    print(f"✓ Dataset created successfully")
    print(f"  - Dataset size: {len(dataset)} samples")
    print(f"  - Mode: {mode}")
    print(f"  - Max microphones: {max_microphones}")
    print(f"  - Sample rate: {sample_rate} Hz")
    
    # Load a sample
    print("\nLoading sample 0...")
    audio, vad, params = dataset[0]
    
    # Print sample information
    print(f"✓ Sample loaded successfully")
    print(f"  - Audio shape: {audio.shape}")
    print(f"  - VAD shape: {vad.shape}")
    print(f"  - VAD speech ratio: {vad.mean():.2%}")
    print(f"  - Number of microphones: {params['num_mics']}")
    print(f"  - Microphone coordinates shape: {params['mic_coords'].shape}")
    print(f"  - Source position shape: {params['source_coords'].shape}")
    print(f"  - Audio duration: {audio.shape[1] / sample_rate:.2f} seconds")
    print(f"  - Audio dtype: {audio.dtype}")
    print(f"\n  Source position (x, y, z): {params['source_coords']}")
    print(f"\n  First microphone position (x, y, z): {params['mic_coords'][0]}")
    print(f"  Audio statistics:")
    print(f"    - Min: {audio.min():.6f}")
    print(f"    - Max: {audio.max():.6f}")
    print(f"    - Mean: {audio.mean():.6f}")
    print(f"    - Std: {audio.std():.6f}")
    
    print("\n✓ All tests passed!")
        
    # 3D Visualization of microphone array and source position
    print("\\nGenerating 3D visualization...")
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions (convert to numpy for matplotlib)
    mic_coords = params['mic_coords'].cpu().numpy()
    source_pos = params['source_coords'].cpu().numpy()
    source_pos = np.array([
            source_pos[0] * math.sin(source_pos[2]) * math.cos(source_pos[1]),
            source_pos[0] * math.sin(source_pos[2]) * math.sin(source_pos[1]),
            source_pos[0] * math.cos(source_pos[2])
        ])
    
    # Plot microphones as blue dots
    ax.scatter(mic_coords[:, 0], mic_coords[:, 1], mic_coords[:, 2], 
                c='blue', marker='o', s=100, label='Microphones', alpha=0.6)
    
    # Plot source as red star
    ax.scatter(source_pos[0], source_pos[1], source_pos[2], 
                c='red', marker='*', s=500, label='Source', alpha=0.9)
    
    # Draw lines from each microphone to source to show spatial relationships
    for i, mic in enumerate(mic_coords):
        ax.plot([mic[0], source_pos[0]], 
                [mic[1], source_pos[1]], 
                [mic[2], source_pos[2]], 
                'gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Label each microphone
    for i, mic in enumerate(mic_coords):
        ax.text(mic[0], mic[1], mic[2], f'  M{i+1}', fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=10)
    ax.set_ylabel('Y (meters)', fontsize=10)
    ax.set_zlabel('Z (meters)', fontsize=10)
    ax.set_title(f'Microphone Array Configuration\\n{params["num_mics"]} Microphones', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization
    # Include both microphones and source in range calculation
    all_points = np.vstack([mic_coords, source_pos.reshape(1, 3)])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ 3D visualization displayed")
    
    # VAD Visualization
    print("\nGenerating VAD visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Audio waveform (first channel)
    audio_np = audio[0].cpu().numpy()  # First mic channel
    time_samples = np.arange(len(audio_np)) / sample_rate
    ax1.plot(time_samples, audio_np, color='blue', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Audio Waveform (First Microphone)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: VAD signal
    vad_np = vad[0, :].cpu().numpy()
    time_frames = np.linspace(0, audio.shape[1] / sample_rate, len(vad_np))
    ax2.fill_between(time_frames, 0, vad_np, color='green', alpha=0.6, label='Voice Activity')
    ax2.plot(time_frames, vad_np, color='darkgreen', linewidth=1.5)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('VAD (1=Speech, 0=Silence)', fontsize=11)
    ax2.set_title(f'Voice Activity Detection (Speech Ratio: {vad.mean():.2%})', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ VAD visualization displayed")
