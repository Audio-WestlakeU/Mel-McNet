import json
import os
from os.path import *
import random
from pathlib import Path
from typing import *

import numpy as np
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import (rank_zero_info, rank_zero_warn)
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample_poly

from data_loaders.utils.collate_func import default_collate_func
from data_loaders.utils.mix import *
from data_loaders.utils.my_distributed_sampler import MyDistributedSampler
from data_loaders.utils.diffuse_noise import (gen_desired_spatial_coherence, gen_diffuse_noise)
from data_loaders.utils.window import reverberation_time_shortening_window


class CHIMEUsingDNSDataset(Dataset):

    def __init__(
        self,
        dns_speech_dir: str,
        rir_dir: str,
        dataset: str,
        target: Optional[str] = None,
        noise_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        noisy_dir: Optional[str] = None,
        ref_channel_idx: int = 4,
        snr: Tuple[float, float] = [10, 20],
        audio_time_len: Optional[float] = None,
        sample_rate: int = 16000,
        noise_type: Union[List[Literal['babble', 'white']],
                          List[Literal['real']]] = ['real'],
        training_num_uttrs: Optional[int] = None,
    ) -> None:
        """The CHIMEUsingDNSDataset class is used to load DNS-5 READ-SPEECH with CHIME-3/4 noise.

        Args:
            dns_dir: dir of DNS dataset
            target:  revb_image, direct_path
            dataset: train_si284, cv_dev93, test_eval92
            audio_time_len: cut the audio to `audio_time_len` seconds if given audio_time_len
        """
        super().__init__()
        assert target in ['revb_image', 'direct_path', None] or target.startswith('RTS'), target
        assert dataset in ['train', 'val', 'test'], dataset

        self.dns_speech_dir = Path(dns_speech_dir).expanduser()
        self.noise_dir = Path(noise_dir).expanduser() if noise_dir is not None else None
        self.target_dir = Path(target_dir).expanduser() if target_dir is not None else None
        self.noisy_dir = Path(noisy_dir).expanduser() if noisy_dir is not None else None
        self.target = target
        self.dataset = dataset
        self.audio_time_len = audio_time_len
        self.sample_rate = sample_rate
        self.noise_type = noise_type
        self.ref_channel_idx = ref_channel_idx
        assert sample_rate == 16000, ('Not implemented for sample rate ', sample_rate)

        self.snr = snr

        # load speech signal and noise
        self.uttrs = [str(u) for u in list(self.dns_speech_dir.rglob('*.wav'))]
        self.uttrs.sort()
        self.training_num_uttrs = training_num_uttrs if training_num_uttrs is not None else len(self.uttrs)

        self.rir_dir = Path(rir_dir).expanduser() / {"train": "train", "val": 'validation', 'test': 'test'}[dataset]
        self.rirs = [str(r) for r in list(Path(self.rir_dir).expanduser().rglob('*.npz'))]
        self.rirs.sort()
        # load & save diffuse parameters
        diffuse_paras_path = (Path(rir_dir) / 'diffuse.npz').expanduser()
        if diffuse_paras_path.exists():
            self.Cs = np.load(diffuse_paras_path, allow_pickle=True)['Cs']
        else:
            pos_mics = np.load(self.rirs[0], allow_pickle=True)['pos_rcv']
            _, self.Cs = gen_desired_spatial_coherence(pos_mics=pos_mics, fs=self.sample_rate, noise_field='spherical', c=343, nfft=512)
            try:
                np.savez(diffuse_paras_path, Cs=self.Cs)
            except:
                ...
        assert len(self.rirs) > 0, f"{str(self.rir_dir)} is empty or not exists"

        # self.length = 35000 if self.dataset.startswith('train') else len(self.uttrs)
        # temporally set for all train data (560h)
        # self.length = len(self.uttrs)
        self.shuffle_rir = True if self.dataset.startswith('train') else False

        if self.noisy_dir is not None and self.target_dir is not None:
            self.use_mixed_data = True
            self.uttrs = [str(u) for u in list(self.noisy_dir.rglob('*.wav'))]
            self.uttrs.sort()
        else:
            self.use_mixed_data = False

        if self.noise_dir is not None:
            if 'real' in self.noise_type:
                self.noises = [str(n) for n in list(self.noise_dir.rglob('*.wav'))]
                self.noises.sort()
            elif 'babble' in self.noise_type:
                self.noises = self.uttrs
        else:
            self.noises = None
        self.length = min(self.training_num_uttrs, len(self.uttrs)) if self.dataset.startswith('train') else len(self.uttrs)

    def __getitem__(self, index_seed: Tuple[int, int]):
        # for each item, an index and seed are given. The seed is used to reproduce this dataset on any machines
        index, seed = index_seed

        rng = np.random.default_rng(np.random.PCG64(seed))

        idx = index % self.length

        if not self.use_mixed_data:
            if self.target is not None:
                # step 1: load single channel clean speech signals
                cleans, sr_src = sf.read(self.uttrs[idx], dtype='float32',always_2d=False) # shape [T]
                if sr_src != self.sample_rate:
                    cleans = resample_poly(cleans, up=self.sample_rate, down=sr_src, axis=0)

                # step 2: pad signals with zeros if they are shorter than the length needed, then cut them to needed
                start = 0
                if self.audio_time_len is not None:
                    frames = int(self.audio_time_len * self.sample_rate)
                    T = cleans.shape[0]
                    if frames > T:
                        cleans = np.concatenate([cleans, np.zeros(frames - T, dtype=cleans.dtype)], axis=0)
                    else:
                        start = rng.integers(low=0, high=T - frames + 1)
                        cleans = cleans[start:start+frames]
                cleans = cleans.reshape(1, -1)  # shape [1,T]


                # step 3: load rirs
                if self.shuffle_rir:
                    rir_this = self.rirs[rng.integers(low=0, high=len(self.rirs))]
                else:
                    rir_this = self.rirs[index % len(self.rirs)]
                rir_dict = np.load(rir_this)
                sr_rir = rir_dict['fs']
                assert sr_rir == self.sample_rate, (sr_rir, self.sample_rate)

                rir = rir_dict['rir']  # shape [nsrc,nmic,time], here nsrc=1 for only one speaker
                if self.target == 'direct_path':  # read simulated direct-path rir
                    rir_target = rir_dict['rir_dp']  # shape [nsrc,nmic,time]
                elif self.target == 'revb_image':  # revb_image
                    rir_target = rir  # shape [nsrc,nmic,time]
                elif self.target.startswith('RTS'):  # e.g. RTS_0.1s
                    rts_time = float(self.target.replace('RTS_', '').replace('s', ''))
                    win = reverberation_time_shortening_window(rir=rir, original_T60=rir_dict['RT60'], target_T60=rts_time, sr=self.sample_rate)
                    rir_target = win * rir
                else:
                    raise NotImplementedError('Unknown target: ' + self.target)
                num_mic = rir.shape[1]


                # step 4: convolve rir and clean speech, then place them at right place to satisfy the given overlap types
                rvbts, targets = zip(*[convolve(wav=wav, rir=rir_spk, rir_target=rir_spk_t, ref_channel=self.ref_channel_idx, align=True) for (wav, rir_spk, rir_spk_t) in zip(cleans, rir, rir_target)])
                targets = np.stack(targets, axis=0)  # shape [nsrc,nmic,time]
                mix = np.sum(rvbts, axis=0)

            else:
                # step 1: load multichannel clean speech signals directly
                cleans, sr_src = sf.read(self.uttrs[idx], dtype='float32', always_2d=False)  # shape [T,C]
                if sr_src != self.sample_rate:
                    cleans = resample_poly(cleans, up=self.sample_rate, down=sr_src, axis=0)


                # step 2: pad signals with zeros if they are shorter than the length needed, then cut them to needed
                start = 0
                if self.audio_time_len is not None:
                    frames = int(self.audio_time_len * self.sample_rate)
                    T = cleans.shape[0]
                    if frames > T:
                        cleans = np.concatenate([cleans, np.zeros(shape=(frames - T, cleans.shape[1]), dtype=cleans.dtype)], axis=0)
                    else:
                        start = rng.integers(low=0, high=T - frames + 1)
                        cleans = cleans[start:start+frames, :]

                cleans = cleans.T # shape [C,T]
                mix = cleans.copy()  # shape [C,T]
                targets = np.expand_dims(cleans, axis=0)  # shape [1,C,T]

            # step 5: generate diffused noise or add real noiseand mix with a sampled SNR
            noise_type = self.noise_type[rng.integers(low=0, high=len(self.noise_type))]
            T = int(self.audio_time_len * self.sample_rate) if self.audio_time_len is not None else mix.shape[0]

            if noise_type == 'real':
                noise_path = self.noises[rng.integers(low=0, high=len(self.noises))]
                noise, sr_noise = sf.read(noise_path, dtype='float32', always_2d=False)  # [T,C]
                if sr_noise != self.sample_rate:
                    noise = resample_poly(noise, up=self.sample_rate, down=sr_noise, axis=0)
                noise = pad_or_cut([noise], lens=[T], rng=rng)[0]
                noise = noise.T

            else:
                if noise_type == 'babble':
                    noises = []
                    for i in range(num_mic):
                        noise_i = np.zeros(shape=(self.audio_time_len * self.sample_rate,), dtype=mix.dtype)
                        for j in range(10):
                            noise_path = self.noises[rng.integers(low=0, high=len(self.noises))]
                            noise_ij, sr_noise = sf.read(noise_path, dtype='float32', always_2d=False)  # [T]
                            if sr_noise != self.sample_rate:
                                noise_ij = resample_poly(noise_ij, up=self.sample_rate, down=sr_noise, axis=0)
                            assert noise_ij.ndim == 1
                            noise_i += pad_or_cut([noise_ij], lens=[T], rng=rng)[0]
                        noises.append(noise_i)
                    noise = np.stack(noises, axis=0).reshape(-1)
                elif noise_type == 'white':
                    noise = rng.normal(size=mix.shape[0] * mix.shape[1])
                noise = gen_diffuse_noise(noise=noise, L=T, Cs=self.Cs, nfft=512, rng=rng)  # shape [num_mic, mix_frames]

            snr_this = rng.uniform(low=self.snr[0], high=self.snr[1])
            coeff = cal_coeff_for_adjusting_relative_energy(wav1=mix, wav2=noise, target_dB=snr_this)
            if coeff is None:
                coeff = 1.0
            noise[:, :] *= coeff
            mix[:, :] = mix + noise


        else: # directly load mixed data and its corresponding target
            mix, sr_src = sf.read(self.uttrs[idx], dtype='float32', always_2d=False)  # shape [T,C]
            if sr_src != self.sample_rate:
                mix = resample_poly(mix, up=self.sample_rate, down=sr_src, axis=0)

            target_path = self.target_dir / Path(self.uttrs[idx]).name
            if target_path.exists():
                targets, sr_tgt = sf.read(target_path,dtype='float32', always_2d=False)  # shape [T,C]
                if sr_tgt != self.sample_rate:
                    targets = resample_poly(targets, up=self.sample_rate, down=sr_tgt, axis=0)
                assert targets.shape[0] == mix.shape[0], (targets.shape[0],mix.shape[0])
            else:
                AssertionError(f"Target file {target_path} does not exist")

            # pad signals with zeros if they are shorter than the length needed, then cut them to needed
            start = 0
            if self.audio_time_len is not None:
                frames = int(self.audio_time_len * self.sample_rate)
                T = mix.shape[0]
                if frames > T:
                    mix = np.concatenate([mix, np.zeros(shape=(frames - T, mix.shape[1]), dtype=mix.dtype)], axis=0)
                    targets = np.concatenate([targets, np.zeros(shape=(frames - T, targets.shape[1]), dtype=targets.dtype)], axis=0)
                else:
                    start = rng.integers(low=0, high=T - frames + 1)
                    mix = mix[start:start+frames,:]
                    targets = targets[start:start+frames,:]

            mix = mix.T   # shape [C,T]
            targets = targets.T  # shape [C,T]
            targets = np.expand_dims(targets, axis=0)  # shape [1,C,T]

        if self.dataset == 'train':
            # scale mix and targets to [-0.9, 0.9]
            scale_value = 0.9 / max(np.max(np.abs(mix)), np.max(np.abs(targets)))
            mix[:, :] *= scale_value
            targets[:, :] *= scale_value

        paras = {
            'index': index,
            'seed': seed,
            'wav_name': self.uttrs[idx],
            'target': self.target,
            'sample_rate': self.sample_rate,
            'dataset': 'chime3',
            'noise_type': self.noise_type,
            'audio_time_len': self.audio_time_len,
        }

        return torch.as_tensor(mix, dtype=torch.float32), torch.as_tensor(targets, dtype=torch.float32), paras

    def __len__(self):
        return self.length


class CHIMEUsingDNSDataModule(LightningDataModule):

    def __init__(
        self,
        dns_speech_dir: str = '/data/home/yangyujie/datasets/DNS5/48k/datasets_fullband/clean_fullband/read_speech',  # a dir contains [early, noise, observation, rirs, speech_source, tail, wsj_8k_zeromean]
        rir_dir: str = '~/datasets/chime3_240703_rirs_generated',  # containing train, validation, and test subdirs
        noise_dir: List[Optional[str]] = ['/data/home/yangyujie/datasets/simu-data/training_dataset/noise_segment_all', None, None, None],  # a dir contains noise files
        target_dir: List[Optional[str]] = [None, '/data/home/yangyujie/datasets/simu-data/validation_dataset/clean', '/data/home/yangyujie/datasets/simu-data/test_dataset/clean', '/data/home/yangyujie/datasets/simu-data/test_dataset/clean'],  # a dir contains target files
        noisy_dir: List[Optional[str]] = [None,'/data/home/yangyujie/datasets/simu-data/validation_dataset/noisy','/data/home/yangyujie/datasets/simu-data/test_dataset/noisy', '/data/home/yangyujie/datasets/simu-data/test_dataset/noisy'],  # a dir contains noisy files
        target: str = "direct_path",  # e.g. revb_image, direct_path
        datasets: List[str] = ['train', 'val', 'test', 'test'],  # datasets for train/val/test/predict
        audio_time_len: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = [3.072, 3.072, None, None],  # audio_time_len (seconds) for train/val/test/predict
        snr: Tuple[float, float] = [-5, 10],  # SNR dB
        noise_type: Union[List[Literal['babble', 'white']],List[Literal['real']]] = ['real'],  # the type of noise
        training_num_uttrs: Optional[int] = None,  # the number of utterances for training, if None, use 35000
        batch_size: List[int] = [1, 1],  # batch size for [train, val, {test, predict}]
        ref_channel_idx: int = 4,  # the index of reference channel for RIR simulation
        num_workers: int = 10,
        collate_func_train: Callable = default_collate_func,
        collate_func_val: Callable = default_collate_func,
        collate_func_test: Callable = default_collate_func,
        seeds: Tuple[Optional[int], int, int, int] = [None, 2, 3, 3],  # random seeds for train/val/test/predict sets
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.dns_speech_dir = dns_speech_dir
        self.rir_dir = rir_dir
        self.target = target
        self.noise_dir = noise_dir
        self.target_dir = target_dir
        self.noisy_dir = noisy_dir
        self.datasets = datasets
        self.audio_time_len = audio_time_len
        self.snr = snr
        self.noise_type = noise_type
        self.persistent_workers = persistent_workers
        self.training_num_uttrs = training_num_uttrs

        self.batch_size = batch_size
        while len(self.batch_size) < 4:
            self.batch_size.append(1)

        rank_zero_info("dataset: CHIMEUsingDNSDataset")
        rank_zero_info(f'train/val/test/predict: {self.datasets}')
        rank_zero_info(f'batch size: train/val/test/predict = {self.batch_size}')
        rank_zero_info(f'audio_time_length: train/val/test/predict = {self.audio_time_len}')
        rank_zero_info(f'target: {self.target}')
        # assert self.batch_size_val == 1, "batch size for validation should be 1 as the audios have different length"
        # assert audio_time_len[2] == None, "the length for test set should be None if you want to test ASR performance"

        self.num_workers = num_workers

        self.collate_func = [collate_func_train, collate_func_val, collate_func_test, default_collate_func]

        self.seeds = []
        for seed in seeds:
            self.seeds.append(seed if seed is not None else random.randint(0, 1000000))

        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.current_stage = stage

    def construct_dataloader(self, dataset, noise_dir, target_dir, noisy_dir, audio_time_len, seed, shuffle, batch_size, collate_fn):
        ds = CHIMEUsingDNSDataset(
            dns_speech_dir=self.dns_speech_dir,
            rir_dir=self.rir_dir,
            target=self.target,
            dataset=dataset,
            noise_dir=noise_dir,
            noisy_dir=noisy_dir,
            target_dir=target_dir,
            ref_channel_idx=4,
            snr=self.snr,
            audio_time_len=audio_time_len,
            noise_type=self.noise_type,
            training_num_uttrs=self.training_num_uttrs,
        )

        return DataLoader(
            ds,
            sampler=MyDistributedSampler(ds, seed=seed, shuffle=shuffle),  #
            batch_size=batch_size,  #
            collate_fn=collate_fn,  #
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[0],
            noise_dir=self.noise_dir[0],
            target_dir=self.target_dir[0],
            noisy_dir=self.noisy_dir[0],
            audio_time_len=self.audio_time_len[0],
            seed=self.seeds[0],
            shuffle=True,
            batch_size=self.batch_size[0],
            collate_fn=self.collate_func[0],
        )

    def val_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[1],
            noise_dir=self.noise_dir[1],
            target_dir=self.target_dir[1],
            noisy_dir=self.noisy_dir[1],
            audio_time_len=self.audio_time_len[1],
            seed=self.seeds[1],
            shuffle=False,
            batch_size=self.batch_size[1],
            collate_fn=self.collate_func[1],
        )

    def test_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[2],
            noise_dir=self.noise_dir[2],
            target_dir=self.target_dir[2],
            noisy_dir=self.noisy_dir[2],
            audio_time_len=self.audio_time_len[2],
            seed=self.seeds[2],
            shuffle=False,
            batch_size=self.batch_size[2],
            collate_fn=self.collate_func[2],
        )

    def predict_dataloader(self) -> DataLoader:
        return self.construct_dataloader(
            dataset=self.datasets[3],
            noise_dir=self.noise_dir[3],
            target_dir=self.target_dir[3],
            noisy_dir=self.noisy_dir[3],
            audio_time_len=self.audio_time_len[3],
            seed=self.seeds[3],
            shuffle=False,
            batch_size=self.batch_size[3],
            collate_fn=self.collate_func[3],
        )


if __name__ == '__main__':
    # """python -m data_loaders.sms_wsj_plus"""
    # from jsonargparse import ArgumentParser
    # parser = ArgumentParser("")
    # parser.add_class_arguments(SmsWsjPlusDataModule, nested_key='data')
    # parser.add_argument('--save_dir', type=str, default='dataset')
    # parser.add_argument('--dataset', type=str, default='predict')
    # parser.add_argument('--gen_unprocessed', type=bool, default=True)
    # parser.add_argument('--gen_target', type=bool, default=True)

    # args = parser.parse_args()
    # os.makedirs(args.save_dir, exist_ok=True)

    # if not args.gen_unprocessed and not args.gen_target:
    #     exit()

    # args_dict = args.data
    # args_dict['num_workers'] = 2  # for debuging
    # datamodule = SmsWsjPlusDataModule(**args_dict)
    # datamodule.setup()

    # if args.dataset.startswith('train'):
    #     dataloader = datamodule.train_dataloader()
    # elif args.dataset.startswith('val'):
    #     dataloader = datamodule.val_dataloader()
    # elif args.dataset.startswith('test'):
    #     dataloader = datamodule.test_dataloader()
    # else:
    #     assert args.dataset.startswith('predict'), args.dataset
    #     dataloader = datamodule.predict_dataloader()

    # if type(dataloader) != dict:
    #     dataloaders = {args.dataset: dataloader}
    # else:
    #     dataloaders = dataloader

    # for ds, dataloader in dataloaders.items():

    #     for idx, (noisy, tar, paras) in enumerate(dataloader):
    #         print(f'{idx}/{len(dataloader)}', end=' ')
    #         # if idx > 10:
    #         #     continue
    #         # write target to dir
    #         if args.gen_target and not args.dataset.startswith('predict'):
    #             tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/target").expanduser()
    #             tar_path.mkdir(parents=True, exist_ok=True)
    #             assert np.max(np.abs(tar[0, :, 0, :].numpy())) <= 1
    #             for spk in range(tar.shape[1]):
    #                 sp = tar_path / basename(paras[0]['saveto'][spk])
    #                 if not sp.exists():
    #                     sf.write(sp, tar[0, spk, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

    #         # write unprocessed's 0-th channel
    #         if args.gen_unprocessed:
    #             tar_path = Path(f"{args.save_dir}/{paras[0]['dataset']}/noisy").expanduser()
    #             tar_path.mkdir(parents=True, exist_ok=True)
    #             assert np.max(np.abs(noisy[0, 0, :].numpy())) <= 1
    #             for spk in range(len(paras[0]['saveto'])):
    #                 sp = tar_path / basename(paras[0]['saveto'][spk])
    #                 if not sp.exists():
    #                     sf.write(sp, noisy[0, 0, :].numpy(), samplerate=paras[0]['sample_rate'])

    #         print(noisy.shape, None if args.dataset.startswith('predict') else tar.shape, paras)

    """python -m data_loaders.chime_using_dns_train"""
    dset = CHIMEUsingDNSDataset(
        dns_speech_dir=
        '/data/home/yangyujie/datasets/DNS5/48k/datasets_fullband/clean_fullband/read_speech',
        rir_dir=
        '/data/home/yangyujie/project/NBSS_pmt/dataset/chime3_240703_rirs_generated',
        target='direct_path',
        dataset='val',
        noise_dir=
        None,
        target_dir='/data/home/yangyujie/datasets/simu-data/validation_dataset/clean',
        noisy_dir='/data/home/yangyujie/datasets/simu-data/validation_dataset/noisy',
        ref_channel_idx=4,
        snr=[-5, 10],
        audio_time_len=3.072,
        sample_rate=16000,
        noise_type=['real'],
    )

    for i in range(dset.length):
        dset.__getitem__((i,i))
