import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, base=10000):
        super().__init__()
        self.base = base

        # self.linear = nn.Sequential(
        #     nn.Linear(3, 30),
        #     nn.Tanh(),
        #     nn.Linear(30, 3)
        # )
    
    def forward(self, signal, mic_coordinates):
        """
        signal: [batch_size, num_mics, hidden_size, time_frames]
        mic_coordinates: [batch_size, num_mics, 3] -> (x, y, z)
        """
        # mic_coordinates = self.linear(mic_coordinates)
        x, y, z = mic_coordinates[..., 0], mic_coordinates[..., 1], mic_coordinates[..., 2]

        x_seg = signal[:, :, :42, :]
        y_seg = signal[:, :, 42:84, :]
        z_seg = signal[:, :, 84:, :]

        theta_x = x.unsqueeze(2) * self.base ** (-2 * torch.arange(0, x_seg.shape[2] // 2, device=signal.device) / x_seg.shape[2])
        theta_y = y.unsqueeze(2) * self.base ** (-2 * torch.arange(0, y_seg.shape[2] // 2, device=signal.device) / y_seg.shape[2])
        theta_z = z.unsqueeze(2) * self.base ** (-2 * torch.arange(0, z_seg.shape[2] // 2, device=signal.device) / z_seg.shape[2])

        x_seg_1 = x_seg[:, :, 0::2, :]
        x_seg_2 = x_seg[:, :, 1::2, :]
        y_seg_1 = y_seg[:, :, 0::2, :]
        y_seg_2 = y_seg[:, :, 1::2, :]
        z_seg_1 = z_seg[:, :, 0::2, :]
        z_seg_2 = z_seg[:, :, 1::2, :]

        x_rot1 = x_seg_1 * torch.cos(theta_x.unsqueeze(-1)) - x_seg_2 * torch.sin(theta_x.unsqueeze(-1))
        x_rot2 = x_seg_1 * torch.sin(theta_x.unsqueeze(-1)) + x_seg_2 * torch.cos(theta_x.unsqueeze(-1))
        y_rot1 = y_seg_1 * torch.cos(theta_y.unsqueeze(-1)) - y_seg_2 * torch.sin(theta_y.unsqueeze(-1))
        y_rot2 = y_seg_1 * torch.sin(theta_y.unsqueeze(-1)) + y_seg_2 * torch.cos(theta_y.unsqueeze(-1))
        z_rot1 = z_seg_1 * torch.cos(theta_z.unsqueeze(-1)) - z_seg_2 * torch.sin(theta_z.unsqueeze(-1))
        z_rot2 = z_seg_1 * torch.sin(theta_z.unsqueeze(-1)) + z_seg_2 * torch.cos(theta_z.unsqueeze(-1))

        x_rot = torch.stack([x_rot1, x_rot2], dim=-2).flatten(2, 3)
        y_rot = torch.stack([y_rot1, y_rot2], dim=-2).flatten(2, 3)
        z_rot = torch.stack([z_rot1, z_rot2], dim=-2).flatten(2, 3)
        
        output = torch.cat([x_rot, y_rot, z_rot], dim=2)
        
        return output


if __name__ == "__main__":
    time_frames = 497
    hidden_size = 128
    rope = RoPE(time_frames=time_frames, hidden_size=hidden_size)
    signal = torch.randn(2, 12, hidden_size, time_frames)
    mic_coordinates = torch.randn(2, 12, 3)
    output = rope(signal, mic_coordinates)
    print(output.shape)