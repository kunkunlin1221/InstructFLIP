# from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModelInterface
from .loss import supcon_loss
from .networks import get_model


def binary_func_sep(model, feat, scale, label, UUID, ce_loss):
    indx_0 = (UUID == 0).cpu()
    label = label.float()

    if indx_0.sum().item() > 0:
        logit_0 = model.fc0(feat[indx_0], scale[indx_0]).squeeze()
        cls_loss_0 = ce_loss(logit_0, label[indx_0])
    else:
        logit_0 = []
        cls_loss_0 = torch.zeros(1).cuda()

    return cls_loss_0


class SAFAS(BaseModelInterface):
    def __init__(
        self,
        modality: str = "rgb",
    ):
        super().__init__()
        self._build_model()
        self.ce_loss = nn.BCELoss()
        self.modality = modality

    def _build_model(self):
        self.model = get_model(
            name="ResNet18_lgt",
            max_iter=-1,
            num_classes=2,
            pretrained=True,
            normed_fc=False,
            use_bias=True,
            simsiam=False,
        )

    def forward(self, xs: torch.Tensor):
        *_, logits = self.model(xs, out_type="all")
        return logits

    def forward_train(self, batch: dict):
        blobs_v1 = batch["ssl_img1"]
        blobs_v2 = batch["ssl_img2"]
        blobs = torch.cat([blobs_v1, blobs_v2], dim=0)
        feats, scales = self.model(blobs, out_type="feat")
        labels = torch.cat([batch["label"], batch["label"]], dim=0)
        UUIDs = torch.zeros_like(labels)
        cls_loss = binary_func_sep(self.model, feats, scales, labels, UUIDs, self.ce_loss)
        feat_normed = F.normalize(feats)
        f1, f2 = torch.split(feat_normed, [len(blobs_v1), len(blobs_v2)], dim=0)
        labels, _ = torch.split(labels, [len(blobs_v1), len(blobs_v2)], dim=0)
        UUIDs, _ = torch.split(UUIDs, [len(blobs_v1), len(blobs_v2)], dim=0)
        feat_loss = supcon_loss(
            torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), UUIDs * 10 + labels, temperature=0.1
        )
        loss = cls_loss + 0.1 * feat_loss
        return {"loss": loss}

    @torch.no_grad()
    def forward_test(self, batch: dict):
        logits = self(batch["img"])
        # scores = torch.softmax(logits, dim=-1)[..., 1]
        return {
            "scores": logits.flatten(),
            "logits": logits,
        }

    @torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
