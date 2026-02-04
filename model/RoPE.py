import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, d_model=128, base=10000):
        super().__init__()
        self.d_model = d_model
        self.base = base

        self.freq_proj = nn.Linear(3, d_model // 2, bias=False)
    
    def forward(self, signal, mic_coordinates):
        """
        signal: [batch_size, num_mics, d_model, time_frames]
        mic_coordinates: [batch_size, num_mics, 3] (x, y, z)
        """
        batch_size, num_mics, d_model, time_frames = signal.shape
        
        # 1. Generate Learned Thetas from 3D coordinates
        # theta shape: [batch_size, num_mics, d_model // 2]
        theta = self.freq_proj(mic_coordinates) 
        
        # 2. Expand theta for the time_frames dimension
        # shape: [batch_size, num_mics, d_model // 2, time_frames]
        theta = theta.unsqueeze(-1).expand(-1, -1, -1, time_frames)

        # IMPROVEMENT 2: Joint Rotation (No feature segmenting)
        # We rotate the entire 128-dim vector as a cohesive 3D position signal.
        # Split into pairs for 2D rotations
        x_1 = signal[:, :, 0::2, :] # Even indices
        x_2 = signal[:, :, 1::2, :] # Odd indices

        # 3. Apply the rotation matrix
        # [cos -sin] [x1]
        # [sin  cos] [x2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        rot_1 = x_1 * cos_theta - x_2 * sin_theta
        rot_2 = x_1 * sin_theta + x_2 * cos_theta

        # 4. Recombine pairs back into the original d_model shape
        # We stack and flatten to maintain the [0, 1, 2, 3...] order
        output = torch.stack([rot_1, rot_2], dim=3).flatten(2, 3)
        
        return output


if __name__ == "__main__":
    time_frames = 497
    hidden_size = 128
    rope = RoPE(d_model=hidden_size)
    signal = torch.randn(2, 12, hidden_size, time_frames)
    mic_coordinates = torch.randn(2, 12, 3)
    output = rope(signal, mic_coordinates)
    print(output.shape)
    