import torch
from .FFT import ConvSTFT
from .Channel_invariant_feature_extractor import Channel_invariant_feature_extractor
from .Microphone_positional_encoding import MicrophonePositionalEncoding
from .SpatioTemporal_block import SpatioTemporal_block
from .SpatioTemporal_block_v1 import SpatioTemporal_block_v1
from .Spatial_spectrum_mapping import Spatial_spectrum_mapping
from .util import target_spatial_spectrum, target_spatial_spectrum_v1
from .RoPE import RoPE


class GI_DOAEnet(torch.nn.Module):
    def __init__(self, MPE_type, model_version='v1'):
        """
        Initialize GI-DOAEnet model with version control.
        
        Args:
            MPE_type: Type of microphone positional encoding (used in v1)
            model_version: 'v1' for basic model (MPE + azimuth only) or 'v2' for improved model (RoPE + azimuth + elevation)
        """
        super(GI_DOAEnet, self).__init__()

        fft_len = 512
        self.feature_size = 128
        self.model_version = model_version

        # Common components for both versions
        self.STFT = ConvSTFT(win_len=fft_len,
                            win_inc=128,
                            fft_len=fft_len,
                            vad_threshold=0.6666,
                            win_type='hann')
        
        self.CIFE = Channel_invariant_feature_extractor(init_feature=fft_len+2,
                                                        feature=self.feature_size, 
                                                        kernel_size=3, 
                                                        padding=1, 
                                                        stride=1, 
                                                        dilation_rate=2, 
                                                        num_blocks=4)

        self.MPE = MicrophonePositionalEncoding(feature=self.feature_size,
                                                    MPE_type=MPE_type,
                                                    alpha=7,
                                                    beta=4)
                                                    
        # Version-specific components
        if model_version == 'v1':            
            self.STDPBs = SpatioTemporal_block_v1(n_heads=16,
                                                     num_blocks=4,
                                                     feature_size=self.feature_size,
                                                     rnn_layers=2)
            
            self.SSMBs = Spatial_spectrum_mapping(
                feature=self.feature_size,
                total_degrees=360,
                degree_resolution=1, 
                num_blocks=3,
                kernel_size=3, 
                dilation_rate=2
            )
            
        elif model_version == 'v2':
            # V2: Improved model with RoPE and azimuth + elevation
            self.STDPBs = SpatioTemporal_block_v1(n_heads=16,
                                              num_blocks=4,
                                              feature_size=self.feature_size,
                                              rnn_layers=2)
            
            self.SSMBs_azimuth = Spatial_spectrum_mapping(
                feature=self.feature_size,
                total_degrees=360,
                degree_resolution=1, 
                num_blocks=3,
                kernel_size=3, 
                dilation_rate=2
            )

            self.SSMBs_elevation = Spatial_spectrum_mapping(
                feature=self.feature_size,
                total_degrees=360,
                degree_resolution=1, 
                num_blocks=3,
                kernel_size=3, 
                dilation_rate=2
            )
        else:
            raise ValueError(f"Invalid model_version: {model_version}. Must be 'v1' or 'v2'")

        self.gammas = [2.5, 2.5, 2.5]
    
    def set_gamma(self, gamma: list[float]):
        self.gammas = gamma
    
    def set_model_version(self, version: str):
        """Change the model version dynamically."""
        if version not in ['v1', 'v2']:
            raise ValueError(f"Invalid version: {version}. Must be 'v1' or 'v2'")
        self.model_version = version

    def forward(self, x, mic_coordinate, vad=None, polar_position=None, return_target=False):
        """
        Forward pass that automatically selects the appropriate version based on model_version parameter.
        """
        if self.model_version == 'v1':
            return self.forward_v1(x, mic_coordinate, vad, polar_position, return_target)
        else:  # v2
            return self.forward_v2(x, mic_coordinate, vad, polar_position, return_target)

    def forward_v1(self, x, mic_coordinate, vad=None, polar_position=None, return_target=False):
        """
        V1 Forward pass: Basic model with MPE and azimuth-only estimation.
        
        Returns:
            If return_target=False: x_out (B, DS, Degree, T) - azimuth only
            If return_target=True: x_out, target, vad_framed
        """
        # STFT
        x_stft_r, x_stft_i = self.STFT(x, cplx=True)
        x_stft = torch.cat([x_stft_r, x_stft_i], dim=2)  # B, C, 2F, T
        
        # Channel invariant feature extraction
        x_feature = self.CIFE(x_stft)  # B, C, M, T

        # Microphone positional encoding
        MPE = self.MPE(mic_coordinate)  # B, C, M
    
        # Spatio-temporal dual-path block (V1 with MPE)
        x_spatio_temporal = self.STDPBs(x_feature, MPE)  # B, C, M, T

        # Spectrum mapping (azimuth only)
        x_out = self.SSMBs(x_spatio_temporal)  # B, DS, Degree, T
        
        if return_target:
            vad_framed = self.STFT.get_vad_framed(vad)
            # For v1, we only have azimuth, so we use degree_candidate from SSMBs
            target = target_spatial_spectrum_v1(
                polar_position, 
                vad_framed, 
                self.SSMBs.degree_candidate, 
                self.gammas
            )  # B, DS, Degree, T

            return x_out, target, vad_framed
        else:
            return x_out

    def forward_v2(self, x, mic_coordinate, vad=None, polar_position=None, return_target=False):
        """
        V2 Forward pass: Improved model with RoPE and azimuth + elevation estimation.
        
        Returns:
            If return_target=False: x_out_az, x_out_el
            If return_target=True: x_out_az, x_out_el, target_az, target_el, vad_framed
        """
        # STFT
        x_stft_r, x_stft_i = self.STFT(x, cplx=True)
        x_stft = torch.cat([x_stft_r, x_stft_i], dim=2)  # B, C, 2F, T
        
        # Channel invariant feature extraction
        x_feature = self.CIFE(x_stft)  # B, C, M, T

        # Microphone positional encoding
        MPE = self.MPE(mic_coordinate)  # B, C, M
    
        # Spatio-temporal dual-path block (V2 with RoPE)
        x_spatio_temporal = self.STDPBs(x_feature, MPE)  # B, C, M, T 

        # Spectrum mapping (azimuth and elevation)
        x_out_az = self.SSMBs_azimuth(x_spatio_temporal)  # B, DS, Degree, T
        x_out_el = self.SSMBs_elevation(x_spatio_temporal)  # B, DS, Degree, T
        
        if return_target:
            vad_framed = self.STFT.get_vad_framed(vad)
            target_az, target_el = target_spatial_spectrum(
                polar_position,
                vad_framed,
                self.gammas
            )  # B, DS, Degree, T

            return x_out_az, x_out_el, target_az, target_el, vad_framed
        else:
            return x_out_az, x_out_el