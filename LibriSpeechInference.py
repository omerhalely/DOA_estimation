import os
import torch
from torch.utils.data import DataLoader
from LibriSpeechDataset import LibriSpeechDataset
from model.main import GI_DOAEnet
from tqdm import tqdm
import json

torch.manual_seed(42)


class LibriSpeechInference:
    def __init__(
        self,
        model_name,
        data_path,
        mode,
        fine_tuned,
        device,
        max_microphones,
        sample_rate,
        MPE_type,
        model_version='v1',
        num_microphones=None,
        noise_probability=0.0,
        snr_db=20,
        reverberation_time=None
    ):
        """
        Initialize LibriSpeechInference.
        
        Args:
            data_path: Path to the data directory
            mode: Dataset mode ('train', 'val', 'test')
            device: PyTorch device
            max_microphones: Maximum number of microphones
            sample_rate: Audio sample rate
            MPE_type: Type of microphone positional encoding ('PM', 'FM', etc.)
            model_version: Model version to use ('v1' for azimuth only, 'v2' for azimuth + elevation)
        """
        self.model_name = model_name
        self.data_path = data_path
        self.mode = mode
        self.fine_tuned = fine_tuned
        self.device = device
        self.max_microphones = max_microphones
        self.sample_rate = sample_rate
        self.MPE_type = MPE_type
        self.model_version = model_version
        self.num_microphones = num_microphones

        self.model = self.load_model()

        # Create dataset instance
        self.dataset = LibriSpeechDataset(
            data_path=data_path,
            mode=mode,
            device=device,
            max_microphones=max_microphones,
            sample_rate=sample_rate,
            training_phase=3,
            noise_probability=noise_probability,
            snr_db=snr_db,
            constant_snr=True,
            reverberation=reverberation_time
        )
    
    def load_model(self):
        print(f"Loading model (version: {self.model_version})...")
        if self.model_version == "v1":
            if self.fine_tuned:
                model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.tar")
            else:
                model_path = os.path.join(os.getcwd(), "pretrained", "GI_DOAEnet_{}.tar".format(self.MPE_type))
        elif self.model_version == "v2":
            model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.tar")
        else:
            raise ValueError(f"Invalid model version: {self.model_version}")
        
        pretrained = torch.load(model_path, map_location='cpu')
        if self.model_version == "v2" or self.fine_tuned:
            pretrained = pretrained["model_state_dict"]
        model = GI_DOAEnet(MPE_type=self.MPE_type, model_version=self.model_version)
        model.load_state_dict(pretrained, strict=True)
        model.to(self.device)
        return model
    
    def MAE_azimuth(self, pred, target, vad):
        dist, theta_rad, phi_rad = target[0]
        pred = pred[vad[:, 0, :] == 1]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        error = torch.unsqueeze(torch.abs(pred - torch.rad2deg(theta_rad)), dim=0)
        error = torch.concatenate((error, 360 - error), dim=0)
        return torch.mean(torch.min(error, dim=0).values)
    
    def MAE_elevation(self, pred, target, vad):
        dist, theta_rad, phi_rad = target[0]
        pred = pred[vad[:, 0, :] == 1]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        error = torch.abs(pred - torch.rad2deg(phi_rad))
        return torch.mean(error)

    def __call__(self):
        self.model.eval()

        if self.num_microphones is not None:
            self.dataset.set_num_microphones(self.num_microphones)
        else:
            self.dataset.sample_num_microphones()

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

        total_azimuth_loss = 0
        total_elevation_loss = 0
        
        for batch_idx, (audio, vad, room_params) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Inference ({self.model_version})"):
            if self.num_microphones is not None:
                self.dataset.set_num_microphones(self.num_microphones)
            else:
                self.dataset.sample_num_microphones()
            
            audio = audio.to(self.device)
            vad = vad.to(self.device)
            mic_coordinates = room_params['mic_coords'].to(self.device)
            source_coordinates = room_params['source_coords'].to(self.device)

            with torch.no_grad():
                if self.model_version == 'v1':
                    # V1: Only azimuth output
                    x_out, target, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)
                    
                    # Get peaks for azimuth (last depth scale output)
                    peaks_az, peaks_idx_az = torch.max(x_out[:, -1, :, :], dim=1)
                    
                    # Calculate MAE for azimuth only
                    loss_az = self.MAE_azimuth(peaks_idx_az, source_coordinates, vad_framed)
                    total_azimuth_loss += loss_az.item()
                    
                elif self.model_version == 'v2':
                    # V2: Both azimuth and elevation outputs
                    x_out_az, x_out_el, target_az, target_el, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)
                    
                    # Get peaks for azimuth (last depth scale output)
                    peaks_az, peaks_idx_az = torch.max(x_out_az[:, -1, :, :], dim=1)
                    # Get peaks for elevation (last depth scale output)
                    peaks_el, peaks_idx_el = torch.max(x_out_el[:, -1, :, :], dim=1)
                    peaks_idx_el = peaks_idx_el.float() / 3.0
                    peaks_idx_el += 30

                    # Calculate MAE for azimuth and elevation separately
                    loss_az = self.MAE_azimuth(peaks_idx_az, source_coordinates, vad_framed)
                    loss_el = self.MAE_elevation(peaks_idx_el, source_coordinates, vad_framed)
                    
                    total_azimuth_loss += loss_az.item()
                    total_elevation_loss += loss_el.item()
        
        avg_azimuth_loss = total_azimuth_loss / len(dataloader)
        
        if self.model_version == 'v1':
            # V1: Return only azimuth loss (elevation is None)
            return avg_azimuth_loss, None
        else:
            # V2: Return both azimuth and elevation losses
            avg_elevation_loss = total_elevation_loss / len(dataloader)
            return avg_azimuth_loss, avg_elevation_loss
            
            

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GI-DOAEnet inference on LibriSpeech dataset')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), "data"),
                        help='Path to the data directory (default: ./data)')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'validation', 'test'],
                        help='Dataset mode (default: test)')
    parser.add_argument('--max_microphones', type=int, default=12,
                        help='Maximum number of microphones (default: 12)')
    parser.add_argument('--num_microphones', type=int, default=None,
                        help='Fixed number of microphones (default: None, random sampling)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate in Hz (default: 16000)')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='v1',
                        help='Model name (default: v1)')
    parser.add_argument('--fine_tuned', type=bool, default=False,
                        help='Use fine-tuned model (default: False)')
    parser.add_argument('--model_version', type=str, default='v1', choices=['v1', 'v2'],
                        help='Model version: v1 (azimuth only) or v2 (azimuth + elevation) (default: v1)')
    parser.add_argument('--mpe_type', type=str, default='PM', choices=['PM', 'FM'],
                        help='Microphone Positional Encoding type (default: PM)')
    
    # Output parameters
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output JSON file path (default: results.json)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    
    # Noise parameters
    parser.add_argument('--noise_probability', type=float, default=1.0,
                        help='Probability of adding noise (default: 0.0)')
    parser.add_argument('--snr_db', type=int, default=30,
                        help='Signal-to-noise ratio in dB (default: 30)')
    
    # Room acoustics parameters
    parser.add_argument('--reverberation_time', type=float, default=None,
                        help='Fixed reverberation time (RT60) in seconds (default: None, random between 0.2-1.3s)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    results_path = os.path.join(os.getcwd(), "results", args.output)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"GI-DOAEnet Inference - Configuration")
    print(f"{'='*60}")
    print(f"Data path: {args.data_path}")
    print(f"Mode: {args.mode}")
    print(f"Model version: {args.model_version}")
    print(f"Fine-tuned: {args.fine_tuned}")
    print(f"MPE type: {args.mpe_type}")
    print(f"Max microphones: {args.max_microphones}")
    print(f"Fixed microphones: {args.num_microphones if args.num_microphones else 'Random sampling'}")
    print(f"Noise probability: {args.noise_probability}")
    print(f"SNR: {args.snr_db}")
    print(f"Reverberation time: {args.reverberation_time if args.reverberation_time else 'Random between 0.2-1.3s'}")
    print(f"Device: {device}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Create inference instance
    inference = LibriSpeechInference(
        model_name=args.model_name,
        data_path=args.data_path,
        mode=args.mode,
        fine_tuned=args.fine_tuned,
        device=device,
        max_microphones=args.max_microphones,
        sample_rate=args.sample_rate,
        MPE_type=args.mpe_type,
        model_version=args.model_version,
        num_microphones=args.num_microphones,
        noise_probability=args.noise_probability,
        snr_db=args.snr_db,
        reverberation_time=args.reverberation_time
    )
    
    # Run inference
    loss_az, loss_el = inference()
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Results for Model Version: {args.model_version.upper()} - MPE: {args.mpe_type}")
    print(f"{'='*60}")
    print(f"Average Azimuth MAE: {loss_az:.2f} degrees")
    
    if loss_el is not None:
        # V2 model has elevation
        print(f"Average Elevation MAE: {loss_el:.2f} degrees")
        
        results = {
            "model_version": args.model_version,
            "mpe_type": args.mpe_type,
            "mode": args.mode,
            "max_microphones": args.max_microphones,
            "fixed_microphones": args.num_microphones,
            "azimuth_mae": loss_az,
            "elevation_mae": loss_el,
            "noise_probability": args.noise_probability,
            "snr_db": args.snr_db,
            "reverberation_time": args.reverberation_time
        }
    else:
        # V1 model has only azimuth
        print(f"Elevation: N/A (V1 model only predicts azimuth)")
        
        results = {
            "model_version": args.model_version,
            "mpe_type": args.mpe_type,
            "mode": args.mode,
            "max_microphones": args.max_microphones,
            "fixed_microphones": args.num_microphones,
            "azimuth_mae": loss_az,
            "elevation_mae": None,
            "noise_probability": args.noise_probability,
            "snr_db": args.snr_db,
            "reverberation_time": args.reverberation_time
        }
    
    print(f"{'='*60}\n")
    
    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")

# Losses for regular GI-DOAEnet:
# PM: 4.1 degrees
# FM: 4.5 degrees
