import torch


class Trainer:
    def __init__(self, model, device, max_microphones, sample_rate, MPE_type):
        self.model = model
        self.device = device
        self.max_microphones = max_microphones
        self.sample_rate = sample_rate
        self.MPE_type = MPE_type