import scipy.signal
import numpy as np
import librosa
from noisereduce.plotting import plot_reduction_steps
from tqdm import tqdm
import warnings

def _stft(y, n_fft, hop_length, win_length, use_cuda=False):
    if use_cuda:
        # return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True)
        return _stft_torch(y, n_fft, hop_length, win_length)
    else:
        return librosa.stft(
            y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
        )

def _istft(y, n_fft, hop_length, win_length, use_cuda=False):
    if use_cuda:
        # return librosa.istft(y, hop_length, win_length)
        return _istft_torch(y, n_fft, hop_length, win_length)
    else:
        return librosa.istft(y, hop_length, win_length)

def _stft_torch(y, n_fft, hop_length, win_length):
    y = torch.from_numpy(y)
    yt = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=torch.hann_window(window_length=win_length, periodic=True),
        center=True, return_complex=True, normalized=False)
    return yt.numpy()

def _istft_cuda(y, n_fft, hop_length, win_length):
    torch_ifft = torch.istft(torch.from_numpy(y), n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=torch.hann_window(window_length=win_length, periodic=True))
    return torch_ifft.numpy()

def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def update_pbar(pbar, message):
    """ writes to progress bar
    """
    if pbar is not None:
        pbar.set_description(message)
        pbar.update(1)


def _smoothing_filter(n_grad_freq, n_grad_time):
    """Generates a filter to smooth the mask for the spectrogram

    Arguments:
        n_grad_freq {[type]} -- [how many frequency channels to smooth over with the mask.]
        n_grad_time {[type]} -- [how many time channels to smooth over with the mask.]
    """

    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    return smoothing_filter


def mask_signal(sig_stft, sig_mask):
    """ Reduces amplitude of time/frequency regions of a spectrogram based upon a mask

    Arguments:
        sig_stft {[type]} -- spectrogram of signal
        sig_mask {[type]} -- mask to apply to signal

    Returns:
        sig_stft_amp [type] -- masked signal
    """
    sig_stft_amp = sig_stft * (1 - sig_mask)
    return sig_stft_amp


def convolve_gaussian(sig_mask, smoothing_filter, use_cuda=False):
    """ Convolves a gaussian filter with a mask (or any image)

    Arguments:
        sig_mask {[type]} -- The signal mask
        smoothing_filter {[type]} -- the filter to convolve

    Keyword Arguments:
        use_cuda {bool} -- use tensorflow.signal or scipy.signal (default: {False})
    """
    if use_cuda:
        smoothing_filter = smoothing_filter * (
            (np.shape(smoothing_filter)[1] - 1) / 2 + 1
        )
        smoothing_filter = smoothing_filter[None, None, :, :].astype("float32")
        smoothing_filter = torch.from_numpy(smoothing_filter)
        img = torch.from_numpy(sig_mask[None, None, :, :].astype("float32"))
        padding = [(d-1)//2 for d in smoothing_filter.shape[-2:]]
        convoloved = torch.nn.functional.conv2d(img, smoothing_filter, padding=padding).squeeze().numpy()
        return convoloved 
    else:
        return scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")


def load_torch(verbose=False):
    try:
        globals()['torch'] = __import__('torch')
        if verbose:
            if torch.cuda.is_available():
                print('GPUs available : {}'.format(torch.cuda.current_device()))
        globals()['torchlibrosa'] = __import__('torchlibrosa')
    except:
        warnings.warn('torchlibrosa or torch is not installed, reverting to non-torch backend')
        return False
    return True

def reduce_noise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    pad_clipping=True,
    use_cuda=False,
    verbose=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        pad_clipping (bool): Pad the signals with zeros to ensure that the reconstructed data is equal length to the data
        use_cuda (bool): Use tensorflow as a backend for convolution and fft to speed up computation
        verbose (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    # load tensorflow if you are using it as a backend
    if use_cuda:
        use_cuda = load_torch(verbose)

    if verbose:
        pbar = tqdm(total=7)
    else:
        pbar = None

    update_pbar(pbar, "STFT on noise")
    # STFT over noise
    noise_stft = _stft(
        noise_clip, n_fft, hop_length, win_length, use_cuda=use_cuda
    )
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    update_pbar(pbar, "STFT on signal")
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    # STFT over signal
    update_pbar(pbar, "STFT on signal")

    # pad signal with zeros to avoid extra frames being clipped if desired
    if pad_clipping:
        nsamp = len(audio_clip)
        audio_clip = np.pad(audio_clip, [0, hop_length], mode="constant")

    sig_stft = _stft(
        audio_clip, n_fft, hop_length, win_length, use_cuda=use_cuda
    )
    # spectrogram of signal in dB
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    update_pbar(pbar, "Generate mask")

    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    update_pbar(pbar, "Smooth mask")
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    # convolve the mask with a smoothing filter
    sig_mask = convolve_gaussian(sig_mask, smoothing_filter, use_cuda)

    sig_mask = sig_mask * prop_decrease
    update_pbar(pbar, "Apply mask")
    # mask the signal

    sig_stft_amp = mask_signal(sig_stft, sig_mask)

    update_pbar(pbar, "Recover signal")
    # recover the signal
    recovered_signal = _istft(
        sig_stft_amp, n_fft, hop_length, win_length, use_cuda=use_cuda
    )
    # fix the recovered signal length if padding signal
    if pad_clipping:
        recovered_signal = librosa.util.fix_length(recovered_signal, nsamp)

    recovered_spec = _amp_to_db(
        np.abs(
            _stft(
                recovered_signal,
                n_fft,
                hop_length,
                win_length,
                use_cuda=use_cuda,
            )
        )
    )
    if verbose:
        plot_reduction_steps(
            noise_stft_db,
            mean_freq_noise,
            std_freq_noise,
            noise_thresh,
            smoothing_filter,
            sig_stft_db,
            sig_mask,
            recovered_spec,
        )
    return recovered_signal
