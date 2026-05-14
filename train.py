#!/usr/bin/env python3
"""
Mel-MCNet Training & Inference Entry Point

Usage:
    # Training from scratch
    python train.py fit --config configs/mel-mcnet-final.yaml

    # Resume training from checkpoint
    python train.py fit --config configs/mel-mcnet-final.yaml --ckpt_path checkpoints/last.ckpt

    # Testing
    python train.py test --config configs/mel-mcnet-final.yaml --ckpt_path checkpoints/your_checkpoint.ckpt

    # Prediction / Inference
    python train.py predict --config configs/mel-mcnet-final.yaml --ckpt_path checkpoints/your_checkpoint.ckpt
"""

import os

os.environ["OMP_NUM_THREADS"] = str(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = str(1)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import mkl
mkl.set_num_threads(1)

import pytorch_lightning as pl
from models.SharedTrainer_mel_final import TrainModule, TrainCLI
from models.utils.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback

if __name__ == '__main__':
    cli = TrainCLI(
        TrainModule,
        pl.LightningDataModule,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={'overwrite': True},
        subclass_mode_data=True,
    )
