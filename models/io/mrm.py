import numpy as np
import torch

EPSILON = np.finfo(np.float32).eps

def build_Magnitude_Ratio_Mask(noisy: torch.Tensor, clean: torch.Tensor,is_power:bool = False) -> torch.Tensor:
    """
    noisy: STFT coefficients of noisy speech or power spectrogram of noisy speech;
    clean: STFT coefficients of clean speech or power spectrogram of clean speech;
    is_power: True -> The input should be power spectrogram; False -> The input should be STFT coefficients
    ___________________________________________________________
    Function:
    build the Magnitude Ratio Mask (MRM)
    
    """

    if is_power:
        Mrm = torch.sqrt(clean) / (torch.sqrt(noisy) + EPSILON)
    else:
        if torch.is_complex(clean):
            clean = torch.view_as_real(clean)
        if torch.is_complex(noisy):
            noisy = torch.view_as_real(noisy)

        Mrm = torch.sqrt(torch.sum(torch.square(clean),dim=-1))/(torch.sqrt(torch.sum(torch.square(noisy),dim=-1)) + EPSILON)

    Mrm = torch.clip(Mrm,max=1)

    return Mrm

def build_Power_Ratio_Mask(noisy: torch.Tensor, clean: torch.Tensor,is_power:bool = False) -> torch.Tensor:

    if is_power:
        Prm = clean / (noisy + EPSILON)
    else:
        if torch.is_complex(clean):
            clean = torch.view_as_real(clean)
        if torch.is_complex(noisy):
            noisy = torch.view_as_real(noisy)

        Prm = torch.sum(torch.square(clean), dim=-1) / (torch.sum(torch.square(noisy), dim=-1) + EPSILON)

    Prm = torch.clip(Prm, max=1)

    return Prm