import librosa
import numpy as np
from typing import *
import torch
from torch import Tensor
from matplotlib import pyplot as plt


def mel_transform(Y: Tensor, transform_type: str = None, n_mels: int = 80, sr: int = 16000, eps: float = 1e-10) -> Tensor:
    """
    Y: STFT coefficients, shape [B,T,F], complex value;
    transform_type: [Optional]
    - logmel
    - mel coefficient
    """
    assert len(Y.shape) == 3
    n_fft = 2 * (Y.shape[-1] - 1)
    mel_basis = torch.tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)).to(Y.device)
    if transform_type == 'logmel':
        # n_fft = 2 * (Y.shape[-1] - 1)
        # mel_basis = torch.tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)).to(Y.device)
        # output = torch.log10(torch.clip((torch.abs(Y)**2) @ (mel_basis.T), min=1e-10))
        output = torch.log(torch.clip((torch.abs(Y)**2) @ (mel_basis.T), min=eps))
    elif transform_type == 'mel coefficient':
        # n_fft = 2 * (Y.shape[-1] - 1)
        # mel_basis = torch.tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)).to(Y.device)
        output = torch.view_as_complex(torch.einsum('abcd,ec->abed', torch.view_as_real(Y), mel_basis).contiguous())
    elif transform_type == 'melmag_phase':
        mag = (torch.abs(Y)**2) @ (mel_basis.T)
        phase_index = torch.argmax(mel_basis, dim=1) # shape [n_mels]
        phase_stft = torch.exp(-1j * torch.angle(Y))  # mag=1
        phase_mel = phase_stft[:, :, phase_index]
        output = mag * phase_mel
    elif transform_type == "mel_energy":
        output = (torch.abs(Y)**2) @ (mel_basis.T)
    elif transform_type == "mel_maganitude":
        output = (torch.abs(Y)) @ (mel_basis.T)
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)

    assert output.shape[-1] == n_mels, output.shape[-1]
    return output


def safe_log(x: Tensor, clip_val: float = 1e-7) -> Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))

def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """
    Plot a spectrogram and convert it to a numpy array.

    Args:
        spectrogram (ndarray): Spectrogram data.

    Returns:
        ndarray: Numpy array representing the plotted spectrogram.
    """
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Save a matplotlib figure to a numpy array.

    Args:
        fig (Figure): Matplotlib figure object.

    Returns:
        ndarray: Numpy array representing the figure.
    """
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
