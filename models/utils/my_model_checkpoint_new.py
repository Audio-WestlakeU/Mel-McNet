from typing import Optional, Dict, Any, Union
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from collections import OrderedDict
from copy import deepcopy
import pytorch_lightning


class My_ModelCheckpoint(ModelCheckpoint):

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = True,
        save_top_k: int = 5,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_epochs: Optional[int] = None,
        enable_version_counter: bool = True,
        save_last_n: Optional[int] = 10,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_epochs=every_n_epochs,
        )
        self.save_last_n = save_last_n
        self.last_n_checkpoints: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if len(self.last_n_checkpoints) >= self.save_last_n:
            self.last_n_checkpoints.popitem(last=False)  # Remove the oldest checkpoint
        if len(self.last_n_checkpoints) < self.save_last_n:
            checkpoint = self._get_metric_interpolated_filepath_name(self._monitor_candidates(trainer), trainer)
            self.last_n_checkpoints[checkpoint] = deepcopy(trainer.model.state_dict())

    def _save_last_n_checkpoint_average(self):
        avg_weights = None
        ckpts = list(self.last_n_checkpoints.values())
        for model_weights in ckpts:
            if avg_weights is None:
                avg_weights = deepcopy(model_weights)
            else:
                for key in avg_weights.keys():
                    if isinstance(avg_weights[key], torch.Tensor) and isinstance(model_weights[key], torch.Tensor):
                        if avg_weights[key].dtype == torch.long:  # 检查类型是否为 long
                            avg_weights[key] = avg_weights[key].float()  # 将类型转换为 float
                        avg_weights[key] += model_weights[key].float()

        if avg_weights is not None:
            for key in avg_weights.keys():
                if isinstance(avg_weights[key], torch.Tensor):
                    avg_weights[key] /= len(ckpts)

            self._save_model_weights(avg_weights)

    def on_train_end(self, trainer, pl_module):
        if len(self.last_n_checkpoints) > 0:
            self._save_last_n_checkpoint_average()

    def _save_model_weights(self, avg_weights: Dict[str, torch.Tensor]):
        filepath = os.path.join(self.dirpath, f"average_model_{self.save_last_n}.ckpt")
        avg_weights = {k.replace("module.", ""): v for k, v in avg_weights.items()}
        torch.save({
            'state_dict': avg_weights,
            'pytorch-lightning_version': pytorch_lightning.__version__,
        }, filepath)

    def _get_metric_interpolated_filepath_name(self, monitor_candidates: Dict[str, torch.Tensor], trainer: "pl.Trainer", del_filepath: Optional[str] = None) -> str:
        filepath = super()._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
        return filepath