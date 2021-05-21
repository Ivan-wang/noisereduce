import torch
import scipy
import librosa
import numpy as np

def _smoothing_filter(n_grad_freq, n_grad_time):
    smooth = np.outer(
        np.concatenate([np.linspace(0, 1, n_grad_freq+1, endpoint=False),
                        np.linspace(1, 0, n_grad_freq+2)])[1:-1],
        np.concatenate([np.linspace(0, 1, n_grad_time+1, endpoint=False),
                        np.linspace(1, 0, n_grad_time+2)])[1:-1]
    )
    smooth /= np.sum(smooth)
    return smooth

def _smoothing_filter_torch(n_grad_freq, n_grad_time):
    smooth = torch.outer(
        torch.cat([torch.linspace(0, 1, n_grad_freq+2)[:-1],
                   torch.linspace(1, 0, n_grad_freq+2)])[1:-1],
        torch.cat([torch.linspace(0, 1, n_grad_time+2)[:-1],
                   torch.linspace(1, 0, n_grad_time+2)])[1:-1],
    )
    smooth /= torch.sum(smooth)
    return smooth

def reduce_noise_librosa(audio_clip, noise_clip, n_grad_freq=2, n_grad_time=4,
    n_fft=2048, win_length=2048, hop_length=512, n_std_thres=1.5,
    prop_decrease=1.0, pad_clipping=True):

    noise_stft = librosa.stft(
        y=noise_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    noise_stft_db = librosa.core.amplitude_to_db(
        np.abs(noise_stft), ref=1.0, amin=1e-20, top_db=80
    )
    
    mean_freq_noise = np.mean(noise_stft_db, axis=1) # [freq, ]
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thres

    if pad_clipping:
        nsamp = len(audio_clip)
        audio_clip = np.pad(audio_clip, [0, hop_length], mode='constant') 
    
    audio_stft = librosa.stft(
        y=audio_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    audio_stft_db = librosa.core.amplitude_to_db(
        np.abs(audio_stft), ref=1.0, amin=1e-20, top_db=80
    )
    # print(audio_stft_db.shape) # [freq, time]

    db_thres = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(audio_stft_db)[1],
        axis=0,
    ).T # [time, freq]
    audio_mask = audio_stft_db < db_thres

    smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)
    audio_mask = scipy.signal.fftconvolve(audio_mask, smoothing_filter, mode='same')
    audio_mask = audio_mask * prop_decrease

    audio_stft_amp = audio_stft * (1 - audio_mask)
    audio_reduced = librosa.istft(audio_stft_amp, hop_length=hop_length, win_length=win_length)

    if pad_clipping:
        audio_reduced = librosa.util.fix_length(audio_reduced, nsamp)

    return audio_reduced 

import torchaudio
def reduce_noise_torch(audio_clip, noise_clip, smooth_filter, n_grad_freq=2, n_grad_time=4,
    n_fft=2048, win_length=2048, hop_length=512, n_std_thres=1.5,
    prop_decrease=1.0, pad_clipping=True):

    noise_stft = torch.stft(
        noise_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True,
        window=torch.hann_window(window_length=win_length, periodic=True), return_complex=True
    )

    noise_stft_db = torchaudio.functional.amplitude_to_DB(
        torch.abs(noise_stft), multiplier=20., db_multiplier=0., amin=1e-20, top_db=80
    )
    
    mean_freq_noise = torch.mean(noise_stft_db, axis=1) # [freq, ]
    std_freq_noise = torch.std(noise_stft_db, axis=1)
    noise_thres = mean_freq_noise + std_freq_noise * n_std_thres

    if pad_clipping:
        nsamp = len(audio_clip)
        audio_clip = torch.nn.functional.pad(audio_clip, [0, hop_length], mode='constant') 
    
    audio_stft = torch.stft(
        audio_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True,
        window=torch.hann_window(window_length=win_length, periodic=True), return_complex=True
    )

    audio_stft_db = torchaudio.functional.amplitude_to_DB(
        torch.abs(audio_stft), multiplier=20., db_multiplier=0., amin=1e-20, top_db=80
    )

    db_thres = noise_thres.view(1, -1).T
    audio_mask = (audio_stft_db < db_thres).float()

    # smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)
    # smoothing_filter = torch.from_numpy(smoothing_filter).float()
    # smoothing_filter = _smoothing_filter_torch(n_grad_freq, n_grad_time)
    audio_mask = audio_mask[None, None, :, :]
    # smoothing_filter = smoothing_filter[None, None, :, :]
    padding = [(d-1)//2 for d in smooth_filter.shape[-2:]]
    audio_mask = torch.nn.functional.conv2d(audio_mask, smooth_filter, padding=padding)
    audio_mask = audio_mask.squeeze()
    audio_mask = audio_mask * prop_decrease

    audio_stft_amp = audio_stft * (1 - audio_mask)
    audio_reduced = torch.istft(audio_stft_amp, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=torch.hann_window(window_length=win_length, periodic=True))

    if pad_clipping:
        audio_reduced = audio_reduced[..., :nsamp]

    return audio_reduced 

