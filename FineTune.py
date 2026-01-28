import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from LibriSpeechDataset import LibriSpeechDataset
from model.main import GI_DOAEnet


torch.manual_seed(42)


class Trainer:
    def __init__(self, model_name, data_path, mode, device, max_microphones, sample_rate, MPE_type, batch_size, lr, epochs):
        self.model_name = model_name
        self.data_path = data_path
        self.mode = mode
        self.device = device
        self.max_microphones = max_microphones
        self.sample_rate = sample_rate
        self.MPE_type = MPE_type
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        self.model = self.load_model()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # Create dataset instance
        self.dataset = LibriSpeechDataset(
            data_path=data_path,
            mode=mode,
            device=device,
            max_microphones=max_microphones,
            sample_rate=sample_rate,
        )
    
    def load_model(self):
        print("Loading pretrained model...")
        model_path = os.path.join(os.getcwd(), "pretrained", "GI_DOAEnet_{}.tar".format(self.MPE_type))
        pretrained = torch.load(model_path, map_location='cpu')
        model = GI_DOAEnet(MPE_type=self.MPE_type)
        model.load_state_dict(pretrained, strict=True)  
        model.to(self.device)
        return model
    
    def BCE_loss(self, pred, target, vad):
        # vad shape: [batch, num_mics, time_frames]
        # pred/target shape: [batch, num_outputs (3), features, time_frames] or similar
        
        # Take VAD from first microphone: [batch, time_frames]
        vad_mask = vad[:, 0, :]
        
        # Expand VAD to match pred shape for broadcasting
        # Assuming pred has shape [batch, num_outputs, height, time_frames]
        # Expand to [batch, 1, 1, time_frames]
        vad_mask = vad_mask.unsqueeze(1).unsqueeze(1)
        
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        
        # Calculate loss for each of the 3 model outputs
        num_outputs = pred.shape[1]  # Should be 3
        for i in range(num_outputs):
            # Get the i-th output: [batch, height, time_frames]
            layer_pred = pred[:, i, ...].unsqueeze(1)
            layer_target = target[:, i, ...].unsqueeze(1)
            
            # Apply VAD mask to select only speech frames
            # Expand vad_mask if needed to match layer dimensions
            vad_expanded = vad_mask.expand_as(layer_pred)
            
            masked_pred = layer_pred[vad_expanded == 1]
            masked_target = layer_target[vad_expanded == 1]
            
            # Calculate loss for this output
            loss = criterion(masked_pred, masked_target)
            total_loss += loss
        
        return total_loss

    def train_one_epoch(self, epoch):
        self.model.train()

        self.dataset.sample_num_microphones()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        print_every = len(dataloader) // 5
        total_loss = 0
        
        for batch_idx, (audio, vad, room_params) in enumerate(dataloader):
            self.dataset.sample_num_microphones()
            if batch_idx != 0 and batch_idx % print_every == 0:
                average_loss = total_loss / (batch_idx + 1)
                print(f"| Epoch {epoch} | Average Loss: {average_loss:.4f} |")

            audio = audio.to(self.device)
            vad = vad.to(self.device)
            mic_coordinates = room_params['mic_coords'].to(self.device)
            source_coordinates = room_params['source_coords'].to(self.device)

            self.optimizer.zero_grad()

            output, target, vad_framed = self.model(audio, mic_coordinates, vad, source_coordinates, return_target=True)

            loss = self.BCE_loss(output, target, vad_framed)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def __call__(self):
        print(f"Training {self.model_name} on {self.device}...")

        model_dir = os.path.join(os.getcwd(), "saved_models", self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{self.model_name}.tar")

        min_loss = torch.inf
        for epoch in range(self.epochs):
            print("-" * 40)
            loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

            if loss < min_loss:
                min_loss = loss
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    model_name = "GI_DOAEnet_fine_tuned"
    data_path = os.path.join(os.getcwd(), "data")
    mode = "validation"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_microphones = 12
    sample_rate = 16000
    MPE_type = "PM"
    batch_size = 16
    lr = 1e-4
    epochs = 10

    trainer = Trainer(
        model_name=model_name,
        data_path=data_path,
        mode=mode,
        device=device,
        max_microphones=max_microphones,
        sample_rate=sample_rate,
        MPE_type=MPE_type,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
    )
    
    trainer()
