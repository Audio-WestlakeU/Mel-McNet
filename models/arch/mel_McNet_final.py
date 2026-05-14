import torch.nn as nn
from typing import List, Optional, Tuple, Union, Literal

import torch
from torch import Tensor
from torch.nn import Module
from torchaudio.functional import melscale_fbanks
from utils.extract_mel import mel_transform
from models.arch.base.norm import *


class RNN_FC(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int,
            num_layers: int = 2,
            bidirectional: bool = True,
            act_funcs: Tuple[str, str] = ('SiLU', ''),
            use_FC: bool = True,
            RNN_type: Literal['LSTM', 'GRU'] = 'LSTM',
    ):
        super().__init__()

        # Sequence layer
        if RNN_type == 'LSTM':
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif RNN_type == 'GRU':
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        self.sequence_model.flatten_parameters()

        # Fully connected layer
        self.use_FC = use_FC
        if self.use_FC:
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        self.act_funcs = []
        for act_func in act_funcs:
            if act_func == 'SiLU' or act_func == 'swish':
                self.act_funcs.append(nn.SiLU())
            elif act_func == 'ReLU':
                self.act_funcs.append(nn.ReLU())
            elif act_func == 'Tanh':
                self.act_funcs.append(nn.Tanh())
            elif act_func == 'Sigmoid':
                self.act_funcs.append(nn.Sigmoid())
            elif act_func is None or act_func == '':
                self.act_funcs.append(None)  # type:ignore
            else:
                raise NotImplementedError(f"Not implemented activation function {act_func}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        o, _ = self.sequence_model(x)
        if self.act_funcs[0] is not None:
            o = self.act_funcs[0](o)
        if self.use_FC:
            o = self.fc_output_layer(o)
            if self.act_funcs[1] is not None:
                o = self.act_funcs[1](o)
        return o


class my_linear(nn.Module):
    def __init__(
            self,
            dim_input: int = 12,
            dim_output: int = 64,
            use_bias: bool = True,
            use_BN: bool = False,
            use_LN: bool = False,
            activation: Optional[str] = 'ReLU',
    ):
        super().__init__()
        self.use_BN = use_BN
        self.use_LN = use_LN
        self.act: Optional[nn.Module] = None
        self.linear = nn.Linear(in_features=dim_input, out_features=dim_output, bias=use_bias)
        if use_BN:
            self.bn = nn.BatchNorm1d(num_features=dim_output)
        if use_LN:
            self.ln = nn.LayerNorm(dim_output)
        if activation is not None:
            if activation == 'ReLU':
                self.act = nn.ReLU()
            elif activation == 'Sigmoid':
                self.act = nn.Sigmoid()
            elif activation == 'Tanh':
                self.act = nn.Tanh()
            elif activation == 'LeakyReLU':
                self.act = nn.LeakyReLU()
            elif activation == 'SELU':
                self.act = nn.SELU()
            elif activation == 'ELU':
                self.act = nn.ELU()
            elif activation == 'GELU':
                self.act = nn.GELU()
            elif activation == 'Softplus':
                self.act = nn.Softplus()
            else:
                raise NotImplementedError(f"Not implemented activation {activation}")

    def forward(self, x: Tensor) -> Tensor:
        o = self.linear(x)
        if self.use_BN:
            o = self.bn(o)
        if self.use_LN:
            o = self.ln(o)
        if self.act is not None:
            o = self.act(o)
        return o


class my_stft2mel(nn.Module):
    """
    Frequency-subband module (FSB) with attention_like cross-modulation between
    amplitude and phase embeddings, followed by a time-subband module (TSB).
    Amplitude uses a handcrafted mel-filterbank projection; phase uses a linear projection.
    """

    def __init__(
            self,
            n_mels: int = 80,
            n_embed: int = 64,
            n_freqs: int = 257,
            sample_rate: int = 16000,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            activation_func: List[Optional[str]] = ['ReLU', None],
            n_layers: int = 1,
            use_interation: List[bool] = [True, True],
            information_communication_type: str = 'attention_like',  # only 'attention_like' supported
            mel_scale: Literal['htk', 'slaney'] = 'htk',
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.use_interation = use_interation

        self.mel_fbanks = melscale_fbanks(
            n_freqs=n_freqs, n_mels=n_mels,
            sample_rate=sample_rate, f_min=0, f_max=sample_rate // 2,
            mel_scale=mel_scale,
        )

        if information_communication_type != 'attention_like':
            raise NotImplementedError(f"Only 'attention_like' is supported, got {information_communication_type}")

        amp_f_conv1d_list = []
        phase_f_conv1d_list = []
        amp_info_comm_module_list = []

        for _ in range(n_layers):
            # FSB amplitude branch: Conv1d + ReLU + LN
            if activation_func[0] == 'ReLU':
                amp_f_conv1d_list.append(nn.Sequential(
                    nn.Conv1d(in_channels=n_embed, out_channels=n_embed, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.ReLU(),
                    new_norm('LN', n_embed, True),
                ))
            elif activation_func[0] is None:
                amp_f_conv1d_list.append(nn.Sequential(
                    nn.Conv1d(in_channels=n_embed, out_channels=n_embed, kernel_size=kernel_size, stride=stride, padding=padding),
                    new_norm('LN', n_embed, True),
                ))
            else:
                raise NotImplementedError(f"Not implemented activation {activation_func[0]}")

            # FSB phase branch: Conv1d + LN (no activation)
            if activation_func[1] is not None:
                raise NotImplementedError(f"Only None is supported for phase activation, got {activation_func[1]}")
            phase_f_conv1d_list.append(nn.Sequential(
                nn.Conv1d(in_channels=n_embed, out_channels=n_embed, kernel_size=kernel_size, stride=stride, padding=padding),
                new_norm('LN', n_embed, True),
            ))

            # Attention-like gate: phase modulates amplitude
            if use_interation[0]:
                amp_info_comm_module_list.append(nn.Sequential(
                    nn.Linear(in_features=n_embed, out_features=n_embed),
                    nn.Tanh(),
                ))

        self.amp_f_conv1d_layers = nn.ModuleList(amp_f_conv1d_list)
        self.phase_f_conv1d_layers = nn.ModuleList(phase_f_conv1d_list)
        self.amp_info_comm_module_layers = nn.ModuleList(amp_info_comm_module_list)

        # Phase mel projection: linear across frequency axis
        self.phase_linear = nn.Linear(in_features=n_freqs, out_features=n_mels)

        # TSB: mix amplitude and phase over time
        self.t_conv1d = nn.Conv1d(in_channels=n_embed * 2, out_channels=n_embed, kernel_size=6, stride=1, padding=5)

    def forward(self, amp_in: Tensor, phase_in: Tensor) -> Tensor:
        B, T, F, H = amp_in.shape
        amp_in = amp_in.permute(0, 1, 3, 2).reshape(-1, H, F)   # [B*T, H, F]
        phase_in = phase_in.permute(0, 1, 3, 2).reshape(-1, H, F)

        # FSB: alternating Conv1d + attention-like cross-modulation
        for i in range(self.n_layers):
            amp_output = self.amp_f_conv1d_layers[i](amp_in)
            phase_output = self.phase_f_conv1d_layers[i](phase_in)
            if self.use_interation[0] and len(self.amp_info_comm_module_layers) > i:
                amp_in = amp_output * self.amp_info_comm_module_layers[i](phase_output.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                amp_in = amp_output
            phase_in = phase_output

        amp_in = amp_in.reshape(B, T, self.n_embed, F)
        phase_in = phase_in.reshape(B, T, self.n_embed, F)

        # Mel projection: handcrafted filterbank for amplitude, linear for phase
        amp_out = torch.einsum('bthf,fl->bthl', amp_in, self.mel_fbanks.to(amp_in.device)).permute(0, 1, 3, 2)  # [B,T,F',H]
        phase_out = self.phase_linear(phase_in).permute(0, 1, 3, 2)  # [B,T,F',H]

        # TSB: temporal mixing of concatenated amp+phase
        amp_phase = torch.cat([amp_out, phase_out], dim=-1)  # [B,T,F',2H]
        amp_phase = amp_phase.permute(0, 2, 3, 1).reshape(-1, self.n_embed * 2, T)  # [B*F', 2H, T]
        output = self.t_conv1d(amp_phase)[..., :-5].reshape(B, self.n_mels, self.n_embed, T)
        return output.permute(0, 3, 1, 2)  # [B,T,F',H]


class McNet_separation(nn.Module):
    """
    Single-block mel-domain multi-channel speech separation network.
    Merges the former McNet_separation wrapper and Mel_McNet_block into one class.
    Input amplitude uses magnitude (XMag); input phase uses exp representation (cos/sin of phase angles).
    """

    def __init__(
        self,
        # Feature extraction modules
        amp_feature_extraction: Optional[Module] = None,
        phase_feature_extraction: Optional[Module] = None,
        stft2mel: Optional[Union[Module, List[Module]]] = None,
        # Sequence processing modules
        freq: Optional[Union[Module, List[Module]]] = None,
        narr: Optional[Union[Module, List[Module]]] = None,
        sub: Optional[Union[Module, List[Module]]] = None,
        full: Optional[Union[Module, List[Module]]] = None,
        # Processing order and subband config
        order: List[str] = ['freq', 'narr', 'sub', 'full'],
        sub_freqs: Union[int, Tuple[int, int]] = 2,
        look_past_and_ahead: Tuple[int, int] = (5, 0),
        ref_channel: int = 4,
        # Mel-domain input normalization
        use_mel: bool = False,
        ref_chn_idx: int = 4,
        use_mel_domain_norm: bool = False,
        mel_transfer_type: str = 'logmel',
    ):
        super().__init__()
        self.amp_feature_extraction = amp_feature_extraction
        self.phase_feature_extraction = phase_feature_extraction
        self.stft2mel = stft2mel if not isinstance(stft2mel, List) else nn.Sequential(*stft2mel)
        self.freq = freq if not isinstance(freq, List) else nn.Sequential(*freq)
        self.narr = narr if not isinstance(narr, List) else nn.Sequential(*narr)
        self.sub = sub if not isinstance(sub, List) else nn.Sequential(*sub)
        self.full = full if not isinstance(full, List) else nn.Sequential(*full)
        self.order = order
        self.sub_freqs = sub_freqs
        self.look_past_and_ahead = look_past_and_ahead
        self.ref_channel = ref_channel
        self.use_mel = use_mel
        self.ref_chn_idx = ref_chn_idx
        self.use_mel_domain_norm = use_mel_domain_norm
        self.mel_transfer_type = mel_transfer_type

    def forward(self, X: Tensor) -> Tensor:
        B, F, T, _ = X.shape
        X = X.permute(0, 2, 1, 3)  # [B,T,F,2C]

        # Decompose STFT into magnitude, phase-exp, and ref-channel magnitude
        Xcpx = torch.view_as_complex(X.reshape(B, T, F, -1, 2))  # [B,T,F,C]
        XMag = torch.abs(Xcpx)                                    # [B,T,F,C]
        XPhase = torch.angle(Xcpx)                                # [B,T,F,C]
        XPhase_exp = torch.cat([torch.cos(XPhase), torch.sin(XPhase)], dim=-1)  # [B,T,F,2C]
        XrMag = XMag[:, :, :, self.ref_chn_idx].unsqueeze(-1)    # [B,T,F,1]

        # Optionally replace XrMag with mel-domain magnitude of the reference channel
        if self.use_mel:
            X_ref = Xcpx[:, :, :, self.ref_chn_idx]
            XrMag = mel_transform(Y=X_ref, transform_type="mel_maganitude")  # [B,T,F']
            if self.use_mel_domain_norm:
                XrMag = XrMag - torch.mean(XrMag, dim=-2, keepdim=True)
            XrMag = XrMag.unsqueeze(-1)  # [B,T,F',1]

        # Amplitude and phase feature extraction
        amp_embed: Optional[Tensor] = None
        phase_embed: Optional[Tensor] = None
        if self.amp_feature_extraction is not None:
            amp_embed = self.amp_feature_extraction(XMag)
        if self.phase_feature_extraction is not None:
            phase_embed = self.phase_feature_extraction(XPhase_exp)

        # STFT-to-mel projection via FSB+TSB
        i: Tensor = X
        if self.stft2mel is not None:
            assert amp_embed is not None and phase_embed is not None
            i = self.stft2mel(amp_embed, phase_embed)
            B, T, F, _ = i.shape

        # Sequential processing according to order
        for curr in self.order:
            # Append auxiliary features before non-sub3/full4freq steps
            if not curr.startswith('sub3') and not curr.startswith('full4freq'):
                if curr.endswith('+XrMag'):
                    i = torch.concat([i, XrMag], dim=-1)
                    curr = curr.replace('+XrMag', '')
                elif curr.endswith('+XMag'):
                    i = torch.concat([i, XMag], dim=-1)
                    curr = curr.replace('+XMag', '')
                elif curr.endswith('+X'):
                    i = torch.concat([i, X], dim=-1)
                    curr = curr.replace('+X', '')

            reduce_by_num_freqs = False
            if curr == 'sub_':
                reduce_by_num_freqs = True
                curr = 'sub'

            if curr == 'freq':  # frequency-axis RNN: [B*T, F, H]
                B, T, F, _ = i.shape
                i = self.freq(i.reshape(B * T, F, -1)).reshape(B, T, F, -1)  # type:ignore
            elif curr == 'narr':  # narrowband time-axis RNN: [B*F, T, H]
                i = self.narr(i.permute(0, 2, 1, 3).reshape(B * F, T, -1))  # type:ignore
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)
            elif curr == 'sub':  # subband time-axis RNN with symmetric freq padding
                assert isinstance(self.sub_freqs, int)
                sf = self.sub_freqs
                i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                i = torch.concat([i[:, :, :sf, :], i, i[:, :, -sf:, :]], dim=2)
                i = torch.nn.functional.unfold(i, kernel_size=(sf * 2 + 1, 1))
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                if reduce_by_num_freqs:
                    i = i / (sf * 2 + 1)
                i = self.sub(i).reshape(B, F, T, -1).permute(0, 2, 1, 3)  # type:ignore
            elif curr.startswith('sub3'):  # dual-subband: inner from embedding, outer from XrMag
                B, T, F, _ = i.shape
                assert isinstance(self.sub_freqs, tuple) and len(self.sub_freqs) == 2
                sf0, sf1 = self.sub_freqs
                X_ = XrMag.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1) if curr.endswith('+XrMag') \
                    else X.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                if sf0 != 0:
                    X_ = torch.concat([X_[:, :, :sf0, :], X_, X_[:, :, -sf0:, :]], dim=2)
                    Xsub = torch.nn.functional.unfold(X_, kernel_size=(sf0 * 2 + 1, 1))
                else:
                    Xsub = X_.reshape(B * T, -1, F)
                i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)
                i = torch.concat([i[:, :, :sf1, :], i, i[:, :, -sf1:, :]], dim=2)
                i = torch.nn.functional.unfold(i, kernel_size=(sf1 * 2 + 1, 1))
                i = torch.concat([i, Xsub], dim=1)
                i = self.sub(i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1))  # type:ignore
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)
            elif curr.startswith('full4freq'):  # full-band freq-axis RNN with look-ahead context from XrMag
                i = i.reshape(B * T, F, -1)
                if curr.endswith('+XrMag'):
                    lpa = self.look_past_and_ahead
                    XrMag_ = torch.nn.functional.pad(XrMag.permute(0, 2, 3, 1), pad=lpa, mode='constant', value=0)
                    XrMag_ = torch.nn.functional.unfold(XrMag_.reshape(B * F, 1, -1, 1), kernel_size=(lpa[0] + lpa[1] + 1, 1))
                    XrMag_ = XrMag_.reshape(B, F, -1, T).permute(0, 3, 1, 2).reshape(B * T, F, -1)
                    i = torch.cat([i, XrMag_], dim=-1)
                i = self.full(i).reshape(B, T, F, -1)  # type:ignore
            else:
                assert curr == 'full', curr  # full-band time-axis RNN: [B, T, F*H]
                i = self.full(i.reshape(B, T, -1)).reshape(B, T, F, -1)  # type:ignore

        return i.permute(0, 2, 1, 3)


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python -m models.arch.mel_McNet_final
    # Mirrors version_183 config: 6-ch mic, n_fft=512, n_hop=128, sr=16000
    # Input shape: [B, F, T, 2C] = [1, 257, T, 12]
    B, F, T, C = 1, 257, 384, 6   # T=384 ≈ 3.072s at hop=128

    model = McNet_separation(
        amp_feature_extraction=my_linear(dim_input=C,     dim_output=64, use_bias=True, use_LN=True, activation='ReLU'),
        phase_feature_extraction=my_linear(dim_input=C*2, dim_output=64, use_bias=True, use_LN=True, activation=None),
        stft2mel=my_stft2mel(
            n_mels=80, n_embed=64, n_freqs=F, sample_rate=16000,
            kernel_size=3, stride=1, padding=1,
            activation_func=['ReLU', None], n_layers=3,
            use_interation=[True, False],
            information_communication_type='attention_like',
        ),
        freq=RNN_FC(input_size=64,  output_size=64, hidden_size=128, num_layers=1, bidirectional=True,  act_funcs=('', 'ReLU'),    use_FC=True, RNN_type='LSTM'),
        narr=RNN_FC(input_size=64,  output_size=64, hidden_size=256, num_layers=1, bidirectional=False, act_funcs=('', 'ReLU'),    use_FC=True, RNN_type='LSTM'),
        sub =RNN_FC(input_size=327, output_size=64, hidden_size=384, num_layers=1, bidirectional=False, act_funcs=('', 'ReLU'),    use_FC=True, RNN_type='LSTM'),
        full=RNN_FC(input_size=70,  output_size=1,  hidden_size=128, num_layers=1, bidirectional=True,  act_funcs=('', 'Sigmoid'), use_FC=True, RNN_type='LSTM'),
        order=['freq', 'narr', 'sub3+XrMag', 'full4freq+XrMag'],
        sub_freqs=(3, 2),
        look_past_and_ahead=(5, 0),
        ref_channel=4,
        use_mel=True,
        ref_chn_idx=4,
        use_mel_domain_norm=False,
        mel_transfer_type='mel_energy',
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {total_params:.2f} M")

    # Input: real+imag interleaved, shape [B, F, T, 2C]
    X = torch.randn(B, F, T, C * 2)
    with torch.no_grad():
        out = model(X)
    print(f"Input:  {list(X.shape)}")
    print(f"Output: {list(out.shape)}")   # expected [B, F, T, 1]
