from pathlib import Path
from typing import Union

import lightning as L
import pandas as pd
from easydict import EasyDict as edict
from fire import Fire
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.callbacks import ModelSummary

from src.callbacks import CSVLogger, ModelCheckpoint, ModelSummary
from src.datamodule import DataModule
from src.lightningmodule import LigntningModule
from src.utils.tools import colorstr, get_current_time, load_yaml, save_yaml, set_everything

set_everything()

CURRENT = get_current_time()


def train_one_protocol(
    log_dir: str,
    protocol_name: str,
    method_name: str,
    mode_name: str,
    model_cfg: dict,
    data_cfg: dict,
    dataloader_cfg: dict,
    times: int = 5,
):
    dm = DataModule(data_cfg, dataloader_cfg)

    summary = []
    for i in range(1, times + 1):
        L.seed_everything(i + 1, workers=True)

        # logger
        logger = CSVLogger(
            save_dir=log_dir,
            name=f"{protocol_name}/{method_name}/{CURRENT}",
            version=f"{mode_name}/{i}",
        )
        result_folder = Path(logger.log_dir)
        result_folder.mkdir(parents=True, exist_ok=True)
        print(colorstr(f"log_dir: {result_folder}"))

        # ckpt
        ckpt_dir = result_folder / "ckpt"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="{epoch}_{mHTER:.4f}",
                monitor="mHTER",
                mode="min",
                verbose=True,
            ),
            ModelSummary(max_depth=-1),
        ]
        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=10,
            **model_cfg.trainer,
        )

        lm = LigntningModule(cfg=model_cfg.lm)
        trainer.fit(lm, dm)
        if _get_rank() == 0:
            lm.best_table["times"] = i
            summary.append(lm.best_table.reset_index())
        trainer.strategy.barrier()

    if _get_rank() == 0:
        summary = pd.concat(summary, ignore_index=False).set_index(["dataset", "times", "step"])
        mean_summary = pd.DataFrame([], index=summary.index.get_level_values(0).unique(), columns=summary.columns)
        for name in set(summary.index.get_level_values(0)):
            mean_summary.loc[name] = summary.loc[name].mean().to_numpy().astype(float).round(2)

        log_folder = Path(log_dir, protocol_name, method_name, CURRENT, mode_name)
        summary = summary.reset_index().set_index("dataset")
        summary.to_markdown(log_folder / "summary_table.md")
        summary.to_latex(log_folder / "summary_table.tex", float_format="%.2f")
        mean_summary.index.name = "dataset"
        total_mean = mean_summary.mean()
        mean_summary = mean_summary.T
        mean_summary["mean"] = total_mean
        mean_summary = mean_summary.T
        mean_summary.to_markdown(log_folder / "mean_table.md")
        mean_summary.to_latex(log_folder / "mean_table.tex", float_format="%.2f")
    trainer.strategy.barrier()


def main(
    model_cfg_path: str,
    data_cfg_path: str,
    times: int = 3,
    devices: Union[int, list] = -1,
):
    model_cfg_path = Path(model_cfg_path)
    data_cfg_path = Path(data_cfg_path)
    method_name = model_cfg_path.stem

    model_cfg = load_yaml(model_cfg_path)

    if devices is not None:
        model_cfg["trainer"]["strategy"] = "ddp_find_unused_parameters_true"
        model_cfg["trainer"]["devices"] = devices
    data_cfg = load_yaml(data_cfg_path)

    log_dir = Path("data/results", "rgb")
    protocol_name = data_cfg["protocol_name"]

    # save config
    log_folder = Path(log_dir, protocol_name, method_name, CURRENT)
    log_folder.mkdir(parents=True, exist_ok=True)
    save_yaml(data_cfg, log_folder / "data_cfg.yaml")
    save_yaml(model_cfg, log_folder / "model_cfg.yaml")

    data_cfg = edict(data_cfg)
    model_cfg = edict(model_cfg)

    for mode_name, _data_cfg in data_cfg["data"].items():
        train_one_protocol(
            log_dir=log_dir,
            method_name=method_name,
            protocol_name=protocol_name,
            mode_name=mode_name,
            model_cfg=model_cfg,
            data_cfg=_data_cfg,
            dataloader_cfg=data_cfg["dataloader"],
            times=times,
        )


if __name__ == "__main__":
    Fire(main)
