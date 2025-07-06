from typing import Dict, List

import torch
import torch.nn as nn

from ..instruct_flip import Classifier, ImageEncoder, InstructFLIP


class InstructFLIP_VE(InstructFLIP):
    def _build_model(self, t5_model_name):
        self.visual_encoder = ImageEncoder("ViT-B/16")
        self.cls_layer = Classifier(n_dim=768, n_classes=2)

    def lazy_init(self, gpu_id=0):
        return

    def _forward_image_embeds(self, image) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            content_embeds, _ = self.visual_encoder(image)
        return content_embeds

    def _forward_fusion_losses(
        self,
        content_embeds: torch.Tensor,
        cls_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls_preds = self.cls_layer(content_embeds[:, 0].squeeze(1))
        cls_loss = self.ce_loss(cls_preds.float().squeeze(-1), cls_labels)
        return {"cls_loss": cls_loss}

    @torch.no_grad()
    def _forward_fusion_preds(self, content_embeds: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cls_preds = self.cls_layer(content_embeds[:, 0].squeeze(1)).softmax(dim=-1)[:, 1]
        return {
            "cls_preds": cls_preds,
        }

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        content_image_embeds = self._forward_image_embeds(batch["img"])
        fusion_losses = self._forward_fusion_losses(
            content_image_embeds,
            batch["label"],
        )

        return {
            "loss": fusion_losses["cls_loss"],
            "cls_loss": fusion_losses["cls_loss"],
        }

    @torch.no_grad()
    def generate(
        self,
        batch,
    ):
        content_image_embeds = self._forward_image_embeds(batch["img"])
        fusion_preds = self._forward_fusion_preds(
            content_image_embeds,
        )
        return {
            "scores": fusion_preds["cls_preds"],
        }

    def forward_train(self, batch):
        return self(batch)

    def forward_test(self, batch):
        return self.generate(batch)

    @torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
