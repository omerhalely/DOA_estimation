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
        )
    
    def load_model(self):
        print("Loading model...")
        model_path = os.path.join(os.getcwd(), "pretrained", "GI_DOAEnet_{}.tar".format(self.MPE_type))
        pretrained = torch.load(model_path, map_location='cpu')
        model = GI_DOAEnet(MPE_type=self.MPE_type)
        model.load_state_dict(pretrained, strict=True)  
        model.to(self.device)
        return model
    
    def MAE(self, pred, target, vad):
        dist, theta_rad, phi_rad = target[0]
        pred = pred[vad == 1]
        if pred.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        error = torch.unsqueeze(torch.abs(pred - torch.rad2deg(theta_rad)), dim=0)
        error = torch.concatenate((error, 360 - error), dim=0)
        return torch.mean(torch.min(error, dim=0).values)

    def __call__(self):
        self.model.eval()

        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)

        total_loss = 0
        for batch_idx, (audio, vad, room_params) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference"):
            audio = audio.to(self.device)
            vad = vad.to(self.device)
            mic_coordinates = room_params['mic_coords'].to(self.device)
            source_coordinates = room_params['source_coords'].to(self.device)

            with torch.no_grad():
                output = self.model(audio, mic_coordinates)
            peaks, peaks_idx = torch.max(output[:, -1, :, :], dim=1)

            loss = self.MAE(peaks_idx, source_coordinates, vad)
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
            
            

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

    inference_fm = LibriSpeechInference(
        data_path=data_path,
        mode=mode,
        device=device,
        max_microphones=max_microphones,
        sample_rate=sample_rate,
        MPE_type="FM"
    )

    pm_loss = inference_pm()
    fm_loss = inference_fm()
    
    loss = {
        "PM": pm_loss,
        "FM": fm_loss
    }

    print(f"PM Average loss: {loss['PM']:.2f} degrees")
    print(f"FM Average loss: {loss['FM']:.2f} degrees")

    results_path = os.path.join(os.getcwd(), "results.json")    
    with open(results_path, "w") as f:
        json.dump(loss, f)
