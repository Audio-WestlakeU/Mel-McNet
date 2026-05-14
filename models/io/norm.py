from torch import nn
from torch import Tensor
from typing import *
from models.utils.nets_utils import make_pad_mask

import torch


def cumulative_normalization(original_signal_mag: Tensor, sliding_window_len: int = 192) -> Tensor:
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    eps = 1e-10
    mu = 0
    mu_list = []
    batch_size, frame_num, freq_num = original_signal_mag.shape
    for frame_idx in range(frame_num):
        if frame_idx < sliding_window_len:
            alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
            mu = alp * mu + (1 - alp) * torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
        else:
            current_frame_mu = torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
            mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)

    XrMM = torch.stack(mu_list, dim=-1).permute(0, 2, 1).reshape(batch_size, frame_num, 1, 1) + eps
    return XrMM

def cumulative_normalization_revised(original_signal_mag: Tensor, sliding_window_len: int = 192) -> Tensor:
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    eps = 1e-10
    mu = 0.318  # initialize mu with the global mean of the signal in training set
    mu_list = []
    batch_size, frame_num, freq_num = original_signal_mag.shape
    for frame_idx in range(frame_num):
        current_frame_mu = torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
        mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)

    XrMM = torch.stack(mu_list, dim=-1).permute(0, 2, 1).reshape(batch_size, frame_num, 1, 1) + eps
    return XrMM

def cumulative_normalization_frequency(original_signal_mag: Tensor, sliding_window_len: int = 192) -> Tensor:
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    eps = 1e-10
    batch_size, frame_num, freq_num = original_signal_mag.shape
    mu = torch.zeros(freq_num).cuda()
    mu_list = None
    for frame_idx in range(frame_num):
        if frame_idx < sliding_window_len:
            alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
            mu = alp * mu + (1 - alp) * original_signal_mag[:, frame_idx, :].reshape(batch_size, freq_num)
        else:
            current_frame_mu = original_signal_mag[:, frame_idx, :].reshape(batch_size, freq_num)
            mu = alpha * mu + (1 - alpha) * current_frame_mu

        if mu_list is None:
            mu_list = mu.unsqueeze(-1)
        else:
            mu_list = torch.cat([mu_list,mu.unsqueeze(-1)],dim=-1)
    XrMM = mu_list.permute(0,2,1).reshape(batch_size, frame_num, freq_num, 1) + eps

    return XrMM



class Norm(nn.Module):

    def __init__(self, mode: Literal['utterance', 'frequency', 'cumulative_alpha', 'cumulative_alpha_frequency', 'cumulative_alpha_revised','none']) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, X: Tensor, norm_paras: Any = None, inverse: bool = False) -> Any:
        if not inverse:
            return self.norm(X, norm_paras=norm_paras)
        else:
            return self.inorm(X, norm_paras=norm_paras)

    def norm(self, X: Tensor, norm_paras: Any = None, ref_channel: int = None) -> Tuple[Tensor, Any]:
        """ normalization
        Args:
            X: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization or for the normalization of other X's

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """
        if self.mode == 'none':
            return X, (None, None)

        B, C, F, T = X.shape
        if norm_paras is None:
            Xr = X[:, [ref_channel], :, :].clone()  # [B,1,F,T], complex

            if self.mode == 'frequency':
                XrMM = torch.abs(Xr).mean(dim=3, keepdim=True) + 1e-8  # Xr_magnitude_mean, [B,1,F,1]
            elif self.mode == 'utterance':
                XrMM = torch.abs(Xr).mean(dim=(2, 3), keepdim=True) + 1e-8  # Xr_magnitude_mean, [B,1,1,1]
            elif self.mode == 'cumulative_alpha':
                XrMM = cumulative_normalization(original_signal_mag=torch.abs(Xr.permute(0, 3, 2, 1).reshape(B, T, F)))
                XrMM = XrMM.permute(0, 3, 2, 1)  # Xr_magnitude_mean, [B,1,1,T]
            elif self.mode == 'cumulative_alpha_frequency':
                XrMM = cumulative_normalization_frequency(original_signal_mag=torch.abs(Xr.permute(0, 3, 2, 1).reshape(B, T, F))) #[B,T,F,1]
                XrMM = XrMM.permute(0, 3, 2, 1)  # Xr_magnitude_mean, [B,1,F,T]
            elif self.mode == 'cumulative_alpha_revised':
                XrMM = cumulative_normalization_revised(original_signal_mag=torch.abs(Xr.permute(0, 3, 2, 1).reshape(B, T, F)))
                XrMM = XrMM.permute(0, 3, 2, 1)  # Xr_magnitude_mean, [B,1,1,T]
        else:
            Xr, XrMM = norm_paras
        X[:, :, :, :] /= XrMM
        return X, (Xr, XrMM)

    def inorm(self, X: Tensor, norm_paras: Any) -> Tensor:
        """ inverse normalization
        Args:
            x: [B, Chn, F, T], complex
            norm_paras: the paramters for inverse normalization 

        Returns:
            the normalized tensor and the paramters for inverse normalization
        """

        Xr, XrMM = norm_paras
        return X * XrMM

    def extra_repr(self) -> str:
        return f"{self.mode}"

def utterance_mvn(
    x: torch.Tensor,
    ilens: torch.Tensor = None,
    norm_means: bool = True,
    norm_vars: bool = False,
    eps: float = 1.0e-20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply utterance mean and variance normalization

    Args:
        x: (B, T, D), assumed zero padded
        ilens: (B,)
        norm_means:
        norm_vars:
        eps:

    """
    if ilens is None:
        ilens = x.new_full([x.size(0)], x.size(1))
    ilens_ = ilens.to(x.device, x.dtype).view(-1, *[1 for _ in range(x.dim() - 1)])
    # Zero padding
    if x.requires_grad:
        x = x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)
    else:
        x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
    # mean: (B, 1, D)
    mean = x.sum(dim=1, keepdim=True) / ilens_

    if norm_means:
        x -= mean

        if norm_vars:
            var = x.pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            x = x / std
        return x, ilens
    else:
        if norm_vars:
            y = x - mean
            y.masked_fill_(make_pad_mask(ilens, y, 1), 0.0)
            var = y.pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            x /= std
        return x, ilens
