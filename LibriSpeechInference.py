import os
import torch
from torch.utils.data import DataLoader
from LibriSpeechDataset import LibriSpeechDataset
from model.main import GI_DOAEnet
from tqdm import tqdm
import json

torch.manual_seed(42)


class LibriSpeechInference:
    def __init__(self, data_path, mode, device, max_microphones, sample_rate, MPE_type):
        self.data_path = data_path
        self.mode = mode
        self.device = device
        self.max_microphones = max_microphones
        self.sample_rate = sample_rate
        self.MPE_type = MPE_type

        self.model = self.load_model()

        # Create dataset instance
        self.dataset = LibriSpeechDataset(
            data_path=data_path,
            mode=mode,
            device=device,
            max_microphones=max_microphones,
            sample_rate=sample_rate,
            training_phase=3
        )
    
    def load_model(self):
        print("Loading model...")
        model_path = os.path.join(os.getcwd(), "pretrained", "GI_DOAEnet_{}.tar".format(self.MPE_type))
        # model_path = os.path.join(os.getcwd(), "saved_models", "GI_DOAEnet_fine_tuned", "GI_DOAEnet_fine_tuned.tar")
        pretrained = torch.load(model_path, map_location='cpu')
        model = GI_DOAEnet(MPE_type=self.MPE_type)
        model.load_state_dict(pretrained["model_state_dict"], strict=True)
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

        self.dataset.sample_num_microphones()

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

        total_azimuth_loss = 0
        total_elevation_loss = 0
        
        for batch_idx, (audio, vad, room_params) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference"):
            self.dataset.sample_num_microphones()
            
            audio = audio.to(self.device)
            vad = vad.to(self.device)
            mic_coordinates = room_params['mic_coords'].to(self.device)
            source_coordinates = room_params['source_coords'].to(self.device)

            with torch.no_grad():
                x_out_az, x_out_el, target_az, target_el, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)
            
            # Get peaks for azimuth (last depth scale output)
            peaks_az, peaks_idx_az = torch.max(x_out_az[:, -1, :, :], dim=1)
            # Get peaks for elevation (last depth scale output)
            peaks_el, peaks_idx_el = torch.max(x_out_el[:, -1, :, :], dim=1)
            peaks_idx_el += 30

            # Calculate MAE for azimuth and elevation separately
            loss_az = self.MAE_azimuth(peaks_idx_az, source_coordinates, vad_framed)
            loss_el = self.MAE_elevation(peaks_idx_el, source_coordinates, vad_framed)
            
            total_azimuth_loss += loss_az.item()
            total_elevation_loss += loss_el.item()
        
        avg_azimuth_loss = total_azimuth_loss / len(dataloader)
        avg_elevation_loss = total_elevation_loss / len(dataloader)
        
        return avg_azimuth_loss, avg_elevation_loss
            
            

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data")
    mode = "test"
    max_microphones = 12
    sample_rate = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inference_pm = LibriSpeechInference(
        data_path=data_path,
        mode=mode,
        device=device,
        max_microphones=max_microphones,
        sample_rate=sample_rate,
        MPE_type="PM"
    )

    # inference_fm = LibriSpeechInference(
    #     data_path=data_path,
    #     mode=mode,
    #     device=device,
    #     max_microphones=max_microphones,
    #     sample_rate=sample_rate,
    #     MPE_type="FM"
    # )

    pm_loss_az, pm_loss_el = inference_pm()
    print(f"PM Average Azimuth MAE: {pm_loss_az:.2f} degrees")
    print(f"PM Average Elevation MAE: {pm_loss_el:.2f} degrees")
    print(f"PM Average Combined MAE: {(pm_loss_az + pm_loss_el) / 2:.2f} degrees")
    
    # fm_loss_az, fm_loss_el = inference_fm()
    # print(f"FM Average Azimuth MAE: {fm_loss_az:.2f} degrees")
    # print(f"FM Average Elevation MAE: {fm_loss_el:.2f} degrees")
    
    loss = {
        "PM_azimuth": pm_loss_az,
        "PM_elevation": pm_loss_el,
        "PM_combined": (pm_loss_az + pm_loss_el) / 2,
        # "FM_azimuth": fm_loss_az,
        # "FM_elevation": fm_loss_el,
        # "FM_combined": (fm_loss_az + fm_loss_el) / 2,
    }

    results_path = os.path.join(os.getcwd(), "results.json")    
    with open(results_path, "w") as f:
        json.dump(loss, f)


# Losses for regular GI-DOAEnet:
# PM: 4.1 degrees
# FM: 4.5 degrees