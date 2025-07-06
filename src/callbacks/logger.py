import logging
import os
from typing import Any, Dict, Optional, Set, Union

from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.loggers.csv_logs import CSVLogger as PLCSVLogger
from lightning.pytorch.loggers.csv_logs import \
    ExperimentWriter as PLExperimentWriter
from torch import Tensor
from typing_extensions import override

log = logging.getLogger(__name__)


class ExperimentWriter(PLExperimentWriter):
    r"""Experiment writer for CSVLogger.

    Currently, supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    This logger supports logging to remote filesystems via ``fsspec``. Make sure you have it installed.

    Args:
        log_dir: Directory for the experiment logs

    """

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        self.metrics_keys = [x for x in self.metrics_keys if x not in ["epoch", "step"]]
        self.metrics_keys.insert(0, "step")
        self.metrics_keys.insert(0, "epoch")
        return new_keys

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                value = value.item()
            if value >= 1:
                value = f"{value:.2f}" if isinstance(value, float) else value
            elif value < 1 and value > 1e-3:
                value = f"{value:.4f}" if isinstance(value, float) else value
            else:
                value = f"{value:.2E}" if isinstance(value, float) else value
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = f"{step:07d}"
        self.metrics.append(metrics)


class CSVLogger(PLCSVLogger):

    @ property
    @ override
    @ rank_zero_experiment
    def experiment(self):
        r"""Actual _ExperimentWriter object. To use _ExperimentWriter features in your
        :class:`~lightning.pytorch.core.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment
