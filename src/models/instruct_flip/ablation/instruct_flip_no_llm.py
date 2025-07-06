import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import mmcv
import numpy as np
import torch
import torch.nn as nn
from ..instruct_flip import InstructFLIP, ImageEncoder, FusionModulator, Classifier, CueGenerator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
matplotlib.use('Agg')


class InstructFLIPnoLLM(InstructFLIP):


    def _build_model(self, t5_model_name):
        self.max_txt_len = 12
        # vision branch
        self.visual_encoder = ImageEncoder('ViT-B/16')

        # qformer branch
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.content_Qformer, self.content_query_tokens = self.init_Qformer(num_query_token=32, vision_width=768)
        self.style_Qformer, self.style_query_tokens = self.init_Qformer(num_query_token=32, vision_width=768)
        self.content_Qformer.resize_token_embeddings(len(self.tokenizer))
        self.style_Qformer.resize_token_embeddings(len(self.tokenizer))
        self.content_Qformer.cls = None
        self.style_Qformer.cls = None

        # llm branch
        self.content_head = Classifier(n_dim=768, n_classes=11)
        self.style_head = Classifier(n_dim=768, n_classes=5+3+3)

        # fusion branch
        self.fusion_modulator = FusionModulator(
            content_dim=768,
            context_dim=768,
            heads=16,
            n_layers=1,
        )

        # head branch
        self.cls_layer = Classifier(n_dim=768, n_classes=2)
        self.cue_generator = CueGenerator(n_dim=768)
        self.ssl_branch = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )

    def lazy_init(self, gpu_id=0, dtype=torch.bfloat16):
        return

    def _forward_fusion_losses(
        self,
        content_embeds: torch.Tensor,
        content_queries: torch.Tensor,
        style_queries: torch.Tensor,
        cls_labels: torch.Tensor,
        ssl_embeds1: torch.Tensor = None,
        ssl_embeds2: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            queries = torch.cat([content_queries, style_queries], dim=1)
            fusion_inputs = self.fusion_modulator(content_embeds, queries)
            randoms = torch.randn_like(fusion_inputs) * self.cue_noise_weight * cls_labels[..., None, None]
            randoms[:, 0] = 0   # skip cls token
            fusion_inputs += randoms
            b, l, c = fusion_inputs.shape
            cls_inputs, cue_inputs = fusion_inputs.split([1, l-1], dim=1)
            cls_preds = self.cls_layer(cls_inputs.squeeze(1))
            h, w = int((l - 1) ** 0.5), int((l - 1) ** 0.5)
            cue_inputs = cue_inputs.permute(0, 2, 1).reshape(b, c, h, w)
            cue_preds = self.cue_generator(cue_inputs).squeeze(1)

        cls_loss = self.ce_loss(cls_preds.float().squeeze(-1), cls_labels)
        cue_targets = cls_labels[:, None, None, None].expand_as(cue_preds)
        cue_loss = self.cue_loss(cue_preds.float(), cue_targets.float())

        outputs = {
            'cls_loss': cls_loss,
            'cue_loss': cue_loss,
            'ssl_loss': torch.zeros(1, device=cls_labels.device),
        }
        if ssl_embeds1 is not None and ssl_embeds2 is not None:
            logits_ssl, labels_ssl = self.info_nce_loss(ssl_embeds1, ssl_embeds2)
            outputs['ssl_loss'] = self.ce_loss(logits_ssl, labels_ssl)

        return outputs

    @ torch.no_grad()
    def _forward_fusion_preds(
        self,
        content_embeds: torch.Tensor,
        content_queries: torch.Tensor,
        style_queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            queries = torch.cat([content_queries, style_queries], dim=1)
            fusion_inputs = self.fusion_modulator(content_embeds, queries)
            b, l, c = fusion_inputs.shape
            cls_inputs, cue_inputs = fusion_inputs.split([1, l-1], dim=1)
            cls_preds = self.cls_layer(cls_inputs.squeeze(1)).softmax(dim=-1)[:, 1]
            h, w = int((l - 1) ** 0.5), int((l - 1) ** 0.5)
            cue_inputs = cue_inputs.permute(0, 2, 1).reshape(b, c, h, w)
            cue_preds = self.cue_generator(cue_inputs)
        return {
            'cls_preds': cls_preds,
            'cue_preds': cue_preds,
        }

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        content_image_embeds, style_image_embeds = self._forward_image_embeds(batch['img'])
        content_query_outputs = self._forward_qformer(
            content_image_embeds,
            batch['content_input'],
            self.content_Qformer,
            self.content_query_tokens,
        )
        style_query_outputs = self._forward_qformer(
            style_image_embeds,
            batch['style_input'],
            self.style_Qformer,
            self.style_query_tokens,
        )

        content_preds = self.content_head(content_query_outputs[:, 0])
        style_preds = self.style_head(style_query_outputs[:, 0])

        content_lm_loss = self.ce_loss(content_preds, batch['content_label'])
        style_lm_loss = self.ce_loss(style_preds, batch['style_label'])


        if 'ssl_img1' in batch and 'ssl_img2' in batch:
            ssl_embeds1 = self._forward_ssl_embeds(batch['ssl_img1'])
            ssl_embeds2 = self._forward_ssl_embeds(batch['ssl_img2'])
        else:
            ssl_embeds1 = None
            ssl_embeds2 = None
        fusion_losses = self._forward_fusion_losses(
            content_image_embeds,
            content_query_outputs,
            style_query_outputs,
            batch['label'],
            ssl_embeds1=ssl_embeds1,
            ssl_embeds2=ssl_embeds2,
        )
        total_loss = sum(
            [
                self.loss_weights[0] * fusion_losses['cls_loss'],
                self.loss_weights[1] * fusion_losses['cue_loss'],
                self.loss_weights[2] * fusion_losses['ssl_loss'],
                self.loss_weights[3] * content_lm_loss,
                self.loss_weights[4] * style_lm_loss,
            ]
        )

        return {
            'loss': total_loss,
            'cls_loss': fusion_losses['cls_loss'],
            'cue_loss': fusion_losses['cue_loss'],
            'ssl_loss': fusion_losses['ssl_loss'],
            'content_lm_loss': content_lm_loss,
            'style_lm_loss': style_lm_loss,
        }

    @ torch.no_grad()
    def generate(
        self,
        batch,
        disable_lm: bool = False,
    ):
        content_image_embeds, style_image_embeds = self._forward_image_embeds(batch['img'])
        content_query_outputs = self._forward_qformer(
            content_image_embeds,
            batch['content_input'],
            self.content_Qformer,
            self.content_query_tokens,
        )
        style_query_outputs = self._forward_qformer(
            style_image_embeds,
            batch['style_input'],
            self.style_Qformer,
            self.style_query_tokens,
        )
        fusion_preds = self._forward_fusion_preds(
            content_image_embeds,
            content_query_outputs,
            style_query_outputs,
        )
        outputs = {
            'scores': fusion_preds['cls_preds'],
            'cue_preds': fusion_preds['cue_preds'],
        }
        if not disable_lm:
            outputs['content_output_texts'] = self._forward_lm_outputs(
                content_query_outputs,
                batch['content_input'],
            )
            outputs['style_output_texts'] = self._forward_lm_outputs(
                style_query_outputs,
                batch['style_input'],
            )

        return outputs

    def forward_train(self, batch):
        return self(batch)

    def forward_test(self, batch):
        return self.generate(batch, disable_lm=True)

    @ torch.no_grad()
    def show_detail(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        epoch: int,
        mode: str = 'train',
        logger=None,
        gpu_id: int = 0,
        dataset_name: Optional[str] = None,
    ):
        return
