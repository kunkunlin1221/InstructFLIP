from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from .models import build_lr_scheduler, build_model, build_optimizer
from .utils.statistic import calculate_threshold, get_EER_states, get_HTER_at_thr


class LigntningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(**cfg.model)
        self.automatic_optimization = cfg.automatic_optimization

    def configure_optimizers(self):
        parameters = []
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                parameters.append(param)
        optimizer = build_optimizer(params=parameters, **self.cfg.solver.optimizer)
        if self.cfg.solver.lr_scheduler.cls_name == "OneCycleLR":
            self.cfg.solver.lr_scheduler.update({
                "total_steps": self.trainer.estimated_stepping_batches,
            })
        lr_scheduler = build_lr_scheduler(optimizer=optimizer, **self.cfg.solver.lr_scheduler)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        self.model.lazy_init(self.trainer.local_rank)

    def training_step(self, batch, batch_idx, **kwargs):
        losses = self.model.forward_train(batch)
        loss = losses["loss"]

        if not self.automatic_optimization:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            opt.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            self.clip_gradients(
                opt,
                gradient_clip_val=self.trainer.gradient_clip_val,
                gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
            )
            opt.step()
            scheduler.step()

        to_logger = {f"train_{k}": v for k, v in losses.items()}
        to_logger["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log_dict(to_logger, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        if self.trainer.is_global_zero and batch_idx >= self.trainer.estimated_stepping_batches - 10:
            self.model.show_detail(
                batch,
                batch_idx,
                self.current_epoch,
                mode="train",
                logger=self.logger.experiment,
                gpu_id=self.local_rank,
            )
        self.trainer.strategy.barrier()
        return loss

    def on_validation_epoch_start(self) -> None:
        self.valid_results = {}
        self.table_fpath = Path(self.logger.log_dir) / "metrics.md"
        if self.current_epoch == 0:
            self.metric_table = pd.DataFrame(
                [],
                columns=["step", "dataset", "EER(↓)", "ACC(↑)", "HTER(↓)", "AUC(↑)", "T@F=1%(↑)"],
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0, **kwargs):
        preds = self.model.forward_test(batch)["scores"]
        if self.trainer.is_global_zero and batch_idx < 10:
            self.model.show_detail(
                batch,
                batch_idx,
                self.current_epoch,
                mode="valid",
                logger=self.logger.experiment,
                gpu_id=self.local_rank,
                dataset_name=self.trainer.val_dataloaders[dataloader_idx].dataset.name,
            )
        self.trainer.strategy.barrier()
        labels = batch["label"]
        video_ids = batch["video_id"]
        if dataloader_idx not in self.valid_results:
            self.valid_results[dataloader_idx] = {"preds": [preds], "labels": [labels], "video_ids": [video_ids]}
        else:
            self.valid_results[dataloader_idx]["preds"].append(preds)
            self.valid_results[dataloader_idx]["labels"].append(labels)
            self.valid_results[dataloader_idx]["video_ids"].append(video_ids)

    def sync_dist(self, results: dict):
        results = self.all_gather(results)
        return results

    def on_validation_epoch_end(self) -> None:
        mauc = 0
        meer = 0
        macc = 0
        mhter = 0
        mrate = 0

        if self.trainer.strategy == "ddp":
            self.valid_results = self.sync_dist(self.valid_results)

        if self.trainer.is_global_zero:
            for dataloader_idx, valid_results in self.valid_results.items():
                if self.trainer.strategy == "ddp":
                    preds = [torch.cat(x) for x in valid_results["preds"]]
                    labels = [torch.cat(x) for x in valid_results["labels"]]
                    video_ids = [torch.cat(x) for x in valid_results["video_ids"]]
                else:
                    preds = valid_results["preds"]
                    labels = valid_results["labels"]
                    video_ids = valid_results["video_ids"]
                preds = torch.cat(preds).float().cpu().numpy()
                labels = torch.cat(labels).cpu().numpy()
                video_ids = torch.cat(video_ids).cpu().numpy()

                preds_dict = {}
                labels_dict = {}
                for i, video_id in enumerate(video_ids):
                    if video_id not in preds_dict:
                        preds_dict[video_id] = []
                        labels_dict[video_id] = []
                    preds_dict[video_id].append(preds[i])
                    labels_dict[video_id].append(labels[i])

                preds_dict = {k: np.mean(v).item() for k, v in preds_dict.items()}
                labels_dict = {k: np.mean(v).item() for k, v in labels_dict.items()}

                preds = np.array(list(preds_dict.values()))
                labels = np.array(list(labels_dict.values()))

                auc = roc_auc_score(labels, preds)
                eer, threshold, _, _ = get_EER_states(preds, labels)
                acc = calculate_threshold(preds, labels, threshold)
                hter = get_HTER_at_thr(preds, labels, threshold)

                fpr, tpr, _ = roc_curve(labels, preds)
                tpr_filtered = tpr[fpr <= 1 / 100]
                rate = 0
                if len(tpr_filtered):
                    rate = tpr_filtered[-1]
                name = self.trainer.val_dataloaders[dataloader_idx].dataset.name

                auc *= 100
                eer *= 100
                acc *= 100
                hter *= 100
                rate *= 100

                tmp_table = pd.DataFrame(
                    [[self.global_step, name, eer, acc, hter, auc, rate]],
                    columns=self.metric_table.columns,
                )

                meer += eer
                macc += acc
                mhter += hter
                mauc += auc
                mrate += rate
                self.metric_table = pd.concat([self.metric_table, tmp_table], ignore_index=True)

            mauc /= len(self.valid_results)
            meer /= len(self.valid_results)
            macc /= len(self.valid_results)
            mhter /= len(self.valid_results)
            mrate /= len(self.valid_results)

            mean_table = pd.DataFrame(
                [[self.global_step, "mean", meer, macc, mhter, mauc, mrate]],
                columns=self.metric_table.columns,
            )
            self.metric_table = pd.concat([self.metric_table, mean_table], ignore_index=True).round(2)
            out_table = self.metric_table.set_index(["dataset", "step"])
            out_table = out_table.reset_index().set_index("dataset")
            out_table.to_markdown(self.table_fpath)
        self.trainer.strategy.barrier()
        self.log("mHTER", mhter, prog_bar=True, logger=False, on_step=False, on_epoch=True, rank_zero_only=True)

    def on_fit_end(self) -> None:
        if self.trainer.is_global_zero:
            self.metric_table.set_index(["step", "dataset"], inplace=True)
            best_step, _ = self.metric_table.loc[(slice(None), "mean"), "HTER(↓)"].idxmin()
            dataset_names = [x for x in self.metric_table.index.get_level_values(1).unique() if x != "mean"]
            self.best_table = self.metric_table.loc[(best_step, dataset_names), :]
        self.trainer.strategy.barrier()
