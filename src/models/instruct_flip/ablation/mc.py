from typing import Dict, List

import torch
import torch.nn as nn

from ..instruct_flip import Classifier, ImageEncoder, InstructFLIP


class MultiClassifier(InstructFLIP):

    def _build_model(self, t5_model_name):
        self.visual_encoder = ImageEncoder('ViT-B/16')
        self.cls_layer = Classifier(n_dim=768, n_classes=2)
        self.content_cls = Classifier(n_dim=768, n_classes=11)
        self.style_cls = Classifier(n_dim=768, n_classes=5+3+3)

    def lazy_init(self, gpu_id=0):
        return

    def _forward_head_losses(
        self,
        content_embeds: torch.Tensor,
        style_embeds: torch.Tensor,
        cls_labels: torch.Tensor,
        content_labels: torch.Tensor,
        style_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cls_preds = self.cls_layer(content_embeds[:, 0].squeeze(1))
            content_preds = self.content_cls(content_embeds[:, 0].squeeze(1))
            style_preds = self.style_cls(style_embeds[:, 0].squeeze(1))
        cls_loss = self.ce_loss(cls_preds.float().squeeze(-1), cls_labels)
        content_loss = self.ce_loss(content_preds.float().squeeze(-1), content_labels)
        style_loss = self.ce_loss(style_preds.float().squeeze(-1), style_labels)
        losses = [cls_loss, content_loss, style_loss]
        total_loss = 0
        for w, loss in zip(self.loss_weights, losses):
            total_loss += w * loss

        return {
            'cls_loss': cls_loss,
            'content_loss': content_loss,
            'style_loss': style_loss,
            'total_loss': total_loss,
        }

    @ torch.no_grad()
    def _forward_head_preds(self, content_embeds: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            cls_preds = self.cls_layer(content_embeds[:, 0].squeeze(1)).softmax(dim=-1)[:, 1]
        return {
            'cls_preds': cls_preds,
        }

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        content_embeds, style_embeds = self._forward_image_embeds(batch['img'])
        head_losses = self._forward_head_losses(
            content_embeds,
            style_embeds,
            batch['label'],
            batch['content_label'],
            batch['style_label'],
        )
        total_loss = head_losses.pop('total_loss')

        return {
            'loss': total_loss,
            **head_losses
        }

    @ torch.no_grad()
    def generate(self, batch: dict) -> Dict[str, torch.Tensor]:
        content_embeds, _ = self._forward_image_embeds(batch['img'])
        head_preds = self._forward_head_preds(content_embeds)
        return {
            'scores': head_preds['cls_preds'],
        }

    def forward_train(self, batch):
        return self(batch)

    def forward_test(self, batch):
        return self.generate(batch)

    @ torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
