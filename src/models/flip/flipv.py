from typing import Tuple

import clip
import torch
import torch.nn as nn

from ..base import BaseModelInterface
from ..vit.compenents import Classifier, FeatureEmbedder


class FLIPV(BaseModelInterface):
    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int] = (224, 224),
        modality: str = 'rgb',
    ):
        super().__init__()
        self.backbone, _ = clip.load("ViT-B/16", device="cpu")
        x = torch.randn(1, 3, *input_size)
        feat_dim = self.backbone.encode_image(x).shape[-1]
        self.embedder = FeatureEmbedder(feat_dim)
        self.head = Classifier(num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.modality = modality
        if modality != 'rgb':
            raise NotImplementedError(f"Unsupported modality: {modality}")

    def _prepare_blobs(self, batch: dict):
        xs = batch['img']
        return xs

    def forward(self, xs, norm_flag: bool = True):
        feats = self.backbone.encode_image(xs)
        feats = self.embedder(feats, norm_flag=norm_flag)
        cls_outs = self.head(feats, norm_flag=norm_flag)
        return cls_outs, feats

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
