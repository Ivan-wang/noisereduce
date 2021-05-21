import torch
import torch.nn as nn

from .functional import _smoothing_filter_torch
from .functional import reduce_noise_torch

class NoiseReduceModule(nn.Module):
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, n_grad_freq=2,
        n_grad_time=4, n_std_thres=1.5, prop_descrease=1., pad_clipping=True):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.n_grad_freq = n_grad_freq
        self.n_grad_time = n_grad_time

        self.n_std_thres = n_std_thres
        self.prop_decrease = prop_descrease

        self.pad_clipping = pad_clipping

        smooth_filter = _smoothing_filter_torch(self.n_grad_freq, self.n_grad_time)
        smooth_filter = smooth_filter[None, None, :, :]
        self.register_buffer('smooth_filter', smooth_filter)
    
    def forward(self, audio_clip, sr, noise_clip=None):
        if noise_clip is None:
            noise_clip = audio_clip[..., :int(sr*0.5)]
        
        return reduce_noise_torch(audio_clip, noise_clip, self.smooth_filter,
            self.n_grad_freq, self.n_grad_time, self.n_fft, self.win_length, self.hop_length,
            self.n_std_thres, self.prop_decrease, self.pad_clipping)