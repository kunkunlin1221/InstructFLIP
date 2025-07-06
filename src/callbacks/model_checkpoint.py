from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as PLModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary as PLModelSummary


class ModelCheckpoint(PLModelCheckpoint):
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: BaseException) -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        self._save_last_checkpoint(trainer, monitor_candidates)


class ModelSummary(PLModelSummary):

    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        super().__init__(max_depth, **summarize_kwargs)
        self._showed = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._max_depth:
            return
        if not self._showed:
            model_summary = self._summary(trainer, pl_module)
            summary_data = model_summary._get_summary_data()
            total_parameters = model_summary.total_parameters
            trainable_parameters = model_summary.trainable_parameters
            model_size = model_summary.model_size

            if trainer.is_global_zero:
                self.summarize(summary_data, total_parameters, trainable_parameters,
                               model_size, **self._summarize_kwargs)
            trainer.strategy.barrier()
            self._showed = True
