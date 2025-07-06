import timm
import torch
import torch.nn as nn

from ..base import BaseModelInterface
from ..nn.utils import freeze_model


class SimpleClassifier(BaseModelInterface):
    def __init__(self, backbone: dict, num_classes: int, freeze_backbone: bool, modality: str = 'rgb'):
        super().__init__()
        self._build_model(backbone, num_classes, freeze_backbone)
        self.ce_loss = nn.CrossEntropyLoss()
        self.modality = modality

    def _build_model(self, backbone, num_classes, freeze_backbone):
        self.backbone = timm.create_model(**backbone)

        # freeze the vit
        if freeze_backbone:
            freeze_model(self.vit)

        feat_dim = self.backbone.feature_info[-1]['num_chs']
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.head = nn.Linear(in_features=feat_dim, out_features=num_classes)

    def forward(self, *args, **kargs):
        return self.model(*args, **kargs)

    def _prepare_blobs(self, batch: dict):
        xs = batch['img']
        if self.modality == 'mm':
            depths = batch['depth']
            irs = batch['ir']
            xs = torch.concat((xs, depths, irs), dim=1)
        return xs

    def forward_train(self, batch: dict):
        blobs = self._prepare_blobs(batch)
        labels = batch['label']
        logits = self(blobs)
        loss = self.ce_loss(logits, labels)
        return {'loss': loss}

    @torch.no_grad()
    def forward_test(self, batch: dict):
        blobs = self._prepare_blobs(batch)
        logits = self(blobs)
        scores = torch.softmax(logits, dim=-1)[..., 1]
        return {
            'scores': scores,
            'logits': logits,
        }

    @torch.no_grad()
    def show_detail(self, batch, batch_idx, epoch, mode='train', logger=None, gpu_id=0):
        return
