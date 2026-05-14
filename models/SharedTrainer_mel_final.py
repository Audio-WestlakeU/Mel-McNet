from models.utils.base_cli import BaseCLI
# import BaseCLI at the beginning

import os
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = str(1)
os.environ['CUDA LAUNCH BLOCKING'] = '1'
import mkl
mkl.set_num_threads(1)

from typing import *

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from jsonargparse import lazy_instance
from packaging.version import Version
from torch import Tensor
from torchmetrics.functional.audio import permutation_invariant_training as pit
from torchmetrics.functional.audio import pit_permutate
from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BaseFinetuning, BackboneFinetuning
from models.utils.my_model_checkpoint_new import My_ModelCheckpoint
from models.utils.dnsmos import deep_noise_suppression_mean_opinion_score as dnsmos

import models.utils.general_steps as GS
from models.io.loss import *
from models.io.norm import Norm
from models.io.stft import STFT
from models.utils.metrics import (cal_metrics_functional, recover_scale)
from models.utils.base_cli import BaseCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
import data_loaders
from utils.extract_mel import mel_transform, plot_spectrogram_to_numpy
from models.vocos.vocos_online import VOCOS_DECODE
from models.io.mrm import build_Magnitude_Ratio_Mask, build_Power_Ratio_Mask
import soundfile as sf

import torch._dynamo

torch._dynamo.config.suppress_errors = True


class TrainModule(pl.LightningModule):
    """Network Lightning Module, which controls the training, testing, and inference of given arch and io
    """
    name: str  # used by CLI for creating logging dir
    import_path: str = 'models.SharedTrainer_mel_final.TrainModule'

    def __init__(
        self,
        arch: nn.Module,
        channels: List[int],
        ref_channel: int,
        vocoder: VOCOS_DECODE,
        vocoder_ckpt: str = None,
        stft: STFT = STFT(n_fft=256, n_hop=128, win_len=256),
        norm: Norm = Norm(mode='utterance'),
        loss: Loss = Loss(loss_func=neg_si_sdr, pit=True),
        optimizer: Tuple[str, Dict[str, Any]] = ("Adam", {
            "lr": 0.001
        }),
        lr_scheduler: Optional[Tuple[str, Dict[str, Any]]] = ('ReduceLROnPlateau', {
            'mode': 'min',
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-4
        }),
        metrics: List[str] = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'eSTOI'],
        val_metric: str = 'loss',
        write_examples: int = 200,
        ensemble: Union[int, str, List[str], Literal[None]] = None,
        compile: bool = False,
        exp_name: str = "exp",
        use_mel: bool = False,
        mel_transfer_type: str = 'logmel',
        mel_transfer_eps: float = 1e-10,
        use_vocoder: bool = False,
        use_non_log_predict_value: bool = False,
        use_clip: bool = True,
        use_mask: str = None,
    ):
        """
        Args:
            exp_name: set exp_name to notag when debug things. Defaults to "exp".
            metrics: metrics used at test time. Defaults to ['SNR', 'SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ'].
            write_examples: write how many examples at test.
        """

        super().__init__()

        args = locals().copy()  # capture the parameters passed to this function or their edited values

        if compile != False:
            assert Version(torch.__version__) >= Version('2.0.0'), torch.__version__
            self.arch = torch.compile(arch)
        else:
            self.arch = arch

        self.channels = channels
        self.ref_channel = ref_channel
        self.use_mel = use_mel
        self.mel_transfer_type = mel_transfer_type
        self.mel_transfer_eps = mel_transfer_eps
        self.use_vocoder = use_vocoder
        self.use_non_log_predict_value = use_non_log_predict_value
        self.use_clip = use_clip
        self.use_mask = use_mask

        if self.use_vocoder:
            if self.mel_transfer_type == 'logmel' or self.mel_transfer_type == 'mel_energy':
                self.vocoder = vocoder
                save_model = torch.load(vocoder_ckpt, map_location='cpu')
                vocoder_dict = self.vocoder.state_dict()
                vocoder_state_dict = {k: v for k, v in save_model['state_dict'].items() if k in vocoder_dict.keys()}
                vocoder_dict.update(vocoder_state_dict)
                self.vocoder.load_state_dict(vocoder_dict)
                self.vocoder.eval()

        self.stft = stft
        self.norm = norm
        self.loss = loss

        self.val_cpu_metric_input = []
        self.norm_if_exceed_1 = True
        self.name = type(arch).__name__

        # save other parameters to `self`
        for k, v in args.items():
            if k == 'self' or k == '__class__' or hasattr(self, k):
                continue
            setattr(self, k, v)

    def on_train_start(self):
        """Called by PytorchLightning automatically at the start of training"""
        GS.on_train_start(self=self, exp_name=self.exp_name, model_name=self.name, num_chns=max(self.channels) + 1, nfft=self.stft.n_fft, model_class_path=self.import_path)

    def on_fit_start(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B,C,T]

        Returns:
            Tuple[Tensor, Tensor]: ys_hat
        """
        # obtain STFT X
        X, stft_paras = self.stft.stft(x[:, self.channels])  # [B,C,F,T], complex
        B, C, F, T = X.shape
        X, norm_paras = self.norm.norm(X, ref_channel=self.channels.index(self.ref_channel))
        X = X.permute(0, 2, 3, 1)  # B,F,T,C; complex
        X = torch.view_as_real(X).reshape(B, F, T, -1)  # B,F,T,2C

        # network process
        out = self.arch(X)

        out = out.permute(0, 3, 1, 2)  # [B,Spk,F,T]
        Yr_hat = out

        return Yr_hat, norm_paras

    def training_step(self, batch, batch_idx):
        """training step on self.device, called automaticly by PytorchLightning"""
        x, ys, paras = batch  # x: [B,C,T], ys: [B,Spk,C,T]
        yr = ys[:, :, self.ref_channel, :]

        pred, norm_paras = self.forward(x)

        xr, _ = self.stft.stft(x[:, self.ref_channel, :].unsqueeze(1))  # [B,num_spk,F,T], complex
        batch_size, num_spk, num_freq, num_frame = xr.shape
        xr = xr.permute(0, 1, 3, 2)  # [B,num_spk,T,F], complex
        xr = mel_transform(Y=xr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        xr = xr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex
        mask_pred = pred
        yr_hat = torch.square(mask_pred) * xr

        yr, stft_paras = self.stft.stft(yr)  # [B,Spk,F,T], complex
        yr, norm_paras = self.norm.norm(yr, norm_paras=norm_paras)
        batch_size, num_spk, num_freq, num_frame = yr.shape
        yr = yr.permute(0, 1, 3, 2)  # [B,Spk,T,F], complex
        yr = mel_transform(Y=yr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        yr = yr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex

        mask_truth = build_Magnitude_Ratio_Mask(noisy=xr, clean=yr, is_power=True)

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                loss, perms, mask_pred = self.loss(preds=mask_pred, target=mask_truth, reorder=False, reduce_batch=True)
        else:
            loss, perms, mask_pred = self.loss(preds=mask_pred, target=mask_truth, reorder=False, reduce_batch=True)
            if loss.isnan().any():
                print(loss)
                print(paras)

        if self.use_non_log_predict_value:
            if self.use_clip:
                yr_hat = torch.log(torch.clip(yr_hat, min=1e-10))
                yr = torch.log(torch.clip(yr, min=1e-10))

        if self.mel_transfer_type == "logmel" or self.mel_transfer_type == "mel_energy":
            if self.global_step % 1000 == 0 and self.global_rank == 0:
                with torch.no_grad():
                    mel = yr[0, 0]
                    mel_hat = yr_hat[0, 0]
                self.logger.experiment.add_image(
                    "train/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    "train/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                )

        self.log('train/' + self.loss.name, loss, batch_size=ys[0].shape[0], prog_bar=True)
        return loss

    def on_train_epoch_start(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        """validation step on self.device, called automaticly by PytorchLightning"""
        x, ys, paras = batch
        yr = ys[:, :, self.ref_channel, :]

        # forward
        pred, norm_paras = self.forward(x)

        xr, _ = self.stft.stft(x[:, self.ref_channel, :].unsqueeze(1))  # [B,num_spk,F,T], complex
        batch_size, num_spk, num_freq, num_frame = xr.shape
        xr = xr.permute(0, 1, 3, 2)  # [B,num_spk,T,F], complex
        xr = mel_transform(Y=xr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        xr = xr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex
        mask_pred = pred
        yr_hat = torch.square(mask_pred) * xr  # [B,Spk,F,T]

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            # 我也不知道为什么：self.forward放在autocast之后就会出问题，难道是因为lightning内部的GradScaler的原因？
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        yr, stft_paras = self.stft.stft(yr)  # [B,Spk,F,T], complex
        yr, norm_paras = self.norm.norm(yr, norm_paras=norm_paras)
        batch_size, num_spk, num_freq, num_frame = yr.shape
        yr = yr.permute(0, 1, 3, 2)  # [B,Spk,T,F], complex
        yr = mel_transform(Y=yr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        yr = yr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex

        mask_truth = build_Magnitude_Ratio_Mask(noisy=xr, clean=yr, is_power=True)

        # loss
        loss, perms, mask_pred = self.loss(preds=mask_pred, target=mask_truth, reorder=True, reduce_batch=True)

        if self.use_non_log_predict_value:
            if self.use_clip:
                yr_hat = torch.log(torch.clip(yr_hat, min=1e-10))
                yr = torch.log(torch.clip(yr, min=1e-10))

        if self.mel_transfer_type == "logmel" or self.mel_transfer_type == "mel_energy":
            with torch.no_grad():
                mel_loss, _, _ = self.loss(preds=yr_hat, target=yr, reorder=False, reduce_batch=True)
                self.log('val/mel_mse', mel_loss, on_step=True, sync_dist=True, batch_size=ys.shape[0])

            if self.global_rank == 0:
                with torch.no_grad():
                    mel = yr[0, 0]
                    mel_hat = yr_hat[0, 0]

                self.logger.experiment.add_image(
                    "val/mel_target",
                    plot_spectrogram_to_numpy(mel.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                    )
                self.logger.experiment.add_image(
                    "val/mel_pred",
                    plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()),
                    self.global_step,
                    dataformats="HWC",
                    )

        if self.use_vocoder:
            yr = ys[:, :, self.ref_channel, :]
            yr_hat = self.vocoder(yr_hat.reshape(batch_size * num_spk, -1, num_frame), original_len=yr.shape[-1]).reshape(batch_size, num_spk, -1)
            assert yr.shape == yr_hat.shape

            # metrics
            sdr_val = sdr(yr_hat, yr).mean()
            dnsmos_val = dnsmos(yr_hat, paras[0]['sample_rate'], False)
            for idx, name in enumerate(['p808', 'sig', 'bak', 'ovr']):
                self.log(f'val/dnsmos_{name}', dnsmos_val[..., idx].mean().item(), sync_dist=True, batch_size=x.shape[0])
            si_sdr_val = si_sdr(preds=yr_hat, target=yr).mean()

            # logging
            self.log('val/' + self.loss.name, loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=ys.shape[0])
            val_metric = {'loss': loss, 'si_sdr': si_sdr_val, 'sdr': sdr_val}[self.val_metric]
            self.log('val/metric', val_metric, on_step=True, on_epoch=True, sync_dist=True, batch_size=ys.shape[0])  # log val/metric for checkpoint picking

            # always computes the sdr/sisdr for the comparison of different runs
            self.log('val/sdr', sdr_val, on_step=True, on_epoch=True, sync_dist=True, batch_size=ys.shape[0])
            if self.loss.name != 'neg_si_sdr':
                # always computes the neg_si_sdr for the comparison of different runs in Tensorboard
                self.log('val/neg_si_sdr', -si_sdr_val, on_step=True, on_epoch=True, sync_dist=True, batch_size=ys.shape[0])

            # other heavy metrics: pesq
            sample_rate = paras[0]['sample_rate']
            yrs = [[
                ['nb_pesq'] if sample_rate == 8000 else ['nb_pesq', 'wb_pesq'],
                yr_hat.detach().cpu(),
                yr.detach().cpu(),
                None,
                sample_rate,
                'cpu',
            ]]
            self.val_cpu_metric_input += yrs

        else:
            # logging
            self.log('val/' + self.loss.name, loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=ys.shape[0])
            self.log('val/metric', loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=ys.shape[0])  # log val/metric for checkpoint picking

    def on_validation_epoch_end(self) -> None:
        """calculate heavy metrics for every N epochs"""
        GS.on_validation_epoch_end(self=self, cpu_metric_input=self.val_cpu_metric_input, N=5)

    def on_test_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)
        self.results, self.cpu_metric_input = [], []

    def on_test_epoch_end(self):
        GS.on_test_epoch_end(self=self, results=self.results, cpu_metric_input=self.cpu_metric_input, exp_save_path=self.exp_save_path)

    def test_step(self, batch, batch_idx):
        x, ys, paras = batch
        yr = ys[:, :, self.ref_channel, :]
        sample_rate = 16000 if 'sample_rate' not in paras[0] else paras[0]['sample_rate']

        # forward
        pred, norm_paras = self.forward(x)

        xr, _ = self.stft.stft(x[:, self.ref_channel, :].unsqueeze(1))  # [B,num_spk,F,T], complex
        batch_size, num_spk, num_freq, num_frame = xr.shape
        xr = xr.permute(0, 1, 3, 2)  # [B,num_spk,T,F], complex
        xr = mel_transform(Y=xr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        xr = xr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex
        mask_pred = pred
        yr_hat = torch.square(mask_pred) * xr

        if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
            # use float 32 precision for validation and test
            autocast = torch.autocast(device_type=self.device.type, dtype=torch.float32)
            autocast.__enter__()

        yr, stft_paras = self.stft.stft(yr)  # [B,Spk,F,T], complex
        yr, norm_paras = self.norm.norm(yr, norm_paras=norm_paras)
        batch_size, num_spk, num_freq, num_frame = yr.shape
        yr = yr.permute(0, 1, 3, 2)  # [B,Spk,T,F], complex
        yr = mel_transform(Y=yr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        yr = yr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex

        mask_truth = build_Magnitude_Ratio_Mask(noisy=xr, clean=yr, is_power=True)

        # loss
        loss, perms, mask_pred = self.loss(preds=mask_pred, target=mask_truth, reorder=True, reduce_batch=True)
        # print(loss)

        self.log('test/' + self.loss.name, loss, logger=False, batch_size=ys.shape[0])

        if self.use_non_log_predict_value:
            if self.use_clip:
                yr_hat = torch.log(torch.clip(yr_hat, min=self.mel_transfer_eps))
                yr = torch.log(torch.clip(yr, min=self.mel_transfer_eps))

        if self.use_vocoder:
            yr = ys[:, :, self.ref_channel, :]
            yr_hat = self.vocoder(yr_hat.reshape(batch_size * num_spk, -1, num_frame), stft_paras).reshape(batch_size, num_spk, -1)
            assert yr.shape == yr_hat.shape

            # write results & infos
            wavname = os.path.basename(paras[0]['wav_name'])
            result_dict = {'id': batch_idx, 'wavname': wavname, self.loss.name: loss.item()}

            # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
            x_ref = x[:, self.ref_channel, :]
            if self.loss.is_scale_invariant_loss:
                yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

            # calculate metrics, input_metrics, improve_metrics on GPU
            metrics, input_metrics, imp_metrics = cal_metrics_functional(self.metrics, yr_hat[0], yr[0], x_ref.expand_as(yr[0]), sample_rate, device_only='gpu')
            result_dict.update(input_metrics)
            result_dict.update(imp_metrics)
            result_dict.update(metrics)
            self.cpu_metric_input.append((self.metrics, yr_hat[0].detach().cpu(), yr[0].detach().cpu(), x_ref.expand_as(yr[0]).detach().cpu(), sample_rate, 'cpu'))

            # write examples
            if self.write_examples < 0 or paras[0]['index'] < self.write_examples:
                GS.test_setp_write_example(
                    self=self,
                    xr=x[:, self.ref_channel],
                    yr=yr,
                    yr_hat=yr_hat,
                    sample_rate=sample_rate,
                    paras=paras,
                    result_dict=result_dict,
                    wavname=wavname,
                    exp_save_path=self.exp_save_path,
                )

            if self.trainer.precision == '16-mixed' or self.trainer.precision == 'bf16-mixed':
                autocast.__exit__(None, None, None)

            # return metrics, which will be collected, saved in test_epoch_end
            if 'metrics' in paras[0]:
                del paras[0]['metrics']  # remove circular reference
            result_dict['paras'] = paras[0]
            self.results.append(result_dict)
            return result_dict
        else:
            # write results & infos
            wavname = os.path.basename(paras[0]['wav_name'].split('.')[0] + '_SIMU.npy')
            example_dir = os.path.join(self.exp_save_path, 'examples')
            os.makedirs(example_dir, exist_ok=True)
            wav_path = os.path.join(example_dir, wavname)
            np.save(wav_path, yr_hat.reshape(1, yr_hat.shape[-2], yr_hat.shape[-1]).permute(0, 2, 1).cpu().numpy())

    def on_predict_epoch_start(self):
        self.exp_save_path = self.trainer.logger.log_dir
        os.makedirs(self.exp_save_path, exist_ok=True)

    def predict_step(self, batch: Union[Tensor, Tuple[Tensor, Tensor, Dict]], batch_idx: Optional[int] = None, dataloader_idx: Optional[int] = None) -> Tensor:
        """predict step on self.device, could be called dirctly or by PytorchLightning automatically using predict dataset
        Args:
            batch: x or (x, ys, paras). shape of x [B, C, T]

        Returns:
            Tensor: ys_hat, shape [B, Spk, T]
        """
        if isinstance(batch, Tensor):
            x, ys = batch, None
            yr = None
        elif len(batch) == 2:
            x, paras = batch
            ys = None
            yr = None
        else:
            x, ys, paras = batch
            yr = ys[:, :, self.ref_channel, :] if ys is not None else None

        sample_rate = 16000 if 'sample_rate' not in paras else paras['sample_rate']

        # forward
        pred, norm_paras = self.forward(x)

        xr, stft_paras = self.stft.stft(x[:, self.ref_channel, :].unsqueeze(1))  # [B,num_spk,F,T], complex
        batch_size, num_spk, num_freq, num_frame = xr.shape
        xr = xr.permute(0, 1, 3, 2)  # [B,num_spk,T,F], complex
        xr = mel_transform(Y=xr.reshape(batch_size * num_spk, num_frame, num_freq), transform_type=self.mel_transfer_type, eps=self.mel_transfer_eps)
        xr = xr.reshape(batch_size, num_spk, num_frame, -1).permute(0, 1, 3, 2)  # [B,Spk,F',T], complex
        mask_pred = pred
        yr_hat = torch.square(mask_pred) * xr

        if self.use_non_log_predict_value:
            if self.use_clip:
                yr_hat = torch.log(torch.clip(yr_hat, min=self.mel_transfer_eps))

        yr_hat = self.vocoder(yr_hat.reshape(batch_size * num_spk, -1, num_frame), stft_paras).reshape(batch_size, num_spk, -1)

        # write results & infos
        if paras['dataset'][0] == 'chime3':
            wavname = paras['save_to'][0]
        elif paras['dataset'][0] == 'realman':
            wavname = paras[0]['saveto'][0]
        else:
            wavname = os.path.basename(f"{paras[0]['index']}.wav")

        # recover wav's original scale. solve min ||Y^T a - X||F to obtain the scales of the predictions of speakers, cuz sisdr will lose scale
        x_ref = x[:, self.ref_channel, :]
        if self.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(preds=yr_hat, mixture=x_ref, scale_src_together=True if self.loss.loss_func == neg_sa_sdr else False, norm_if_exceed_1=False)

        # write examples
        if self.write_examples < 0 or paras['index'][0] < self.write_examples:
            def write_wav(wav_path: str, wav: torch.Tensor, norm_to: torch.Tensor = None):
                # make sure wav don't have illegal values (abs greater than 1)
                if norm_to:
                    wav = wav / torch.max(torch.abs(wav)) * norm_to
                abs_max_wav = torch.max(torch.abs(wav))
                if abs_max_wav > 1:
                    import warnings
                    warnings.warn(f"abs_max_wav > 1, {abs_max_wav}")
                    wav /= abs_max_wav
                sf.write(wav_path, wav.detach().cpu().numpy(), sample_rate)

            if paras['dataset'][0] == 'chime3':
                wav_path = os.path.join(self.exp_save_path, 'examples', wavname)
                example_dir = os.path.dirname(wav_path)
                os.makedirs(example_dir, exist_ok=True)
                write_wav(wav_path=wav_path, wav=yr_hat[0, 0])
            else:
                assert False, "not implemented"

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        pass

    def configure_optimizers(self):
        """configure optimizer and lr_scheduler"""
        return GS.configure_optimizers(
            self=self,
            optimizer=self.optimizer[0],
            optimizer_kwargs=self.optimizer[1],
            monitor='val/loss',
            lr_scheduler=self.lr_scheduler[0] if self.lr_scheduler is not None else None,
            lr_scheduler_kwargs=self.lr_scheduler[1] if self.lr_scheduler is not None else None,
        )

    def on_after_backward(self: pl.LightningModule):
        if self.trainer.global_rank == 0:
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.logger.experiment.add_scalar(f'grad/{name}', param.grad.norm().item(), self.global_step)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        GS.on_load_checkpoint(self=self, checkpoint=checkpoint, ensemble_opts=self.ensemble, compile=self.compile)


class TrainCLI(BaseCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # # EarlyStopping
        # parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        # early_stopping_defaults = {
        #     "early_stopping.monitor": "val/metric",
        #     "early_stopping.patience": 30,
        #     "early_stopping.mode": "max",
        #     "early_stopping.min_delta": 0.1,
        # }
        # parser.set_defaults(early_stopping_defaults)

        # ModelCheckpoint
        parser.add_lightning_class_args(My_ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_metric{val/metric:.4f}",
            "model_checkpoint.monitor": "val/metric",
            "model_checkpoint.mode": "max",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,  # save all checkpoints
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True,
            # "model_checkpoint.save_last_k": 5
        }
        parser.set_defaults(model_checkpoint_defaults)

        self.add_model_invariant_arguments_to_parser(parser)


if __name__ == '__main__':
    cli = TrainCLI(
        TrainModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )
