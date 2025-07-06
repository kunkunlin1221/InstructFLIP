import os
from typing import Dict, List

import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2 import disabled_train
from lavis.models.blip2_models.modeling_t5 import (T5Config,
                                                   T5ForConditionalGeneration)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, T5TokenizerFast

from ..instruct_flip import (Classifier, FusionModulator, ImageEncoder,
                             InstructFLIP)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InstructFLIP_VE_CB_SB(InstructFLIP):
    def __init__(
        self,
        loss_weights: List[float] = [0.2, 0.4, 0.4],
        t5_model_name: str = "google/flan-t5-base",
    ):
        super().__init__(
            loss_weights=loss_weights,
            t5_model_name=t5_model_name,
        )

    def _build_model(self, t5_model_name):
        # global settings
        self.max_txt_len = 128
        self.max_output_txt_len = 16

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
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name, truncation_side='left')
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name, truncation_side='right')
        t5_config = T5Config.from_pretrained(t5_model_name)
        self.t5_proj = nn.Linear(self.content_Qformer.config.hidden_size, t5_config.hidden_size)
        self.t5_model_name = t5_model_name

        # fusion branch
        self.fusion_modulator = FusionModulator(
            content_dim=768,
            context_dim=768,
            heads=16,
            n_layers=1,
        )

        # head branch
        self.cls_layer = Classifier(n_dim=768, n_classes=2)

    def lazy_init(self, gpu_id=0):
        t5_config = T5Config.from_pretrained(self.t5_model_name)
        t5_config.dense_act_fn = "gelu"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='fp4',
        )
        t5_model = T5ForConditionalGeneration.from_pretrained(
            self.t5_model_name,
            config=t5_config,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map=f"cuda:{gpu_id}",
        )
        # make t5 to qlora model
        t5_model = prepare_model_for_kbit_training(t5_model)
        if self.enable_qlora_on_llm:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            t5_model = get_peft_model(t5_model, lora_config)
        else:
            for p in t5_model.parameters():
                p.requires_grad = False
        self.t5_model = t5_model
        self.t5_model.train = disabled_train

    def _forward_image_embeds(self, image) -> torch.Tensor:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            content_embeds, layer_feats = self.visual_encoder(image)
            style_embeds = []
            for feat in layer_feats:
                avg_feat, std_feat = feat.mean(dim=1), feat.std(dim=1)
                style_embeds.extend([avg_feat, std_feat])
            style_embeds = torch.stack(style_embeds, dim=1)
        return content_embeds, style_embeds

    def _forward_fusion_losses(
        self,
        content_embeds: torch.Tensor,
        content_queries: torch.Tensor,
        style_queries: torch.Tensor,
        cls_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            queries = torch.cat([content_queries, style_queries], dim=1)
            fusion_inputs = self.fusion_modulator(content_embeds, queries)
            cls_preds = self.cls_layer(fusion_inputs[:, 0].squeeze(1))

        cls_loss = self.ce_loss(cls_preds.float().squeeze(-1), cls_labels)
        return {
            'cls_loss': cls_loss,
        }

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
            cls_preds = self.cls_layer(fusion_inputs[:, 0].squeeze(1)).softmax(dim=-1)[:, 1]
        return {
            'cls_preds': cls_preds,
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
        content_inputs = []
        content_outputs = []
        content_query_inds = []
        for i, (inp, oup) in enumerate(zip(batch['content_input'], batch['content_output'])):
            if oup != 'No GT':
                content_inputs.append(inp)
                content_outputs.append(oup)
                content_query_inds.append(i)
        content_lm_loss = self._forward_lm_loss(
            content_query_outputs[content_query_inds],
            content_inputs,
            content_outputs,
        )
        style_inputs = []
        style_outputs = []
        style_query_inds = []
        for i, (inp, oup) in enumerate(zip(batch['style_input'], batch['style_output'])):
            if oup != 'No GT':
                style_inputs.append(inp)
                style_outputs.append(oup)
                style_query_inds.append(i)
        style_lm_loss = self._forward_lm_loss(
            style_query_outputs[style_query_inds],
            style_inputs,
            style_outputs,
        )
        fusion_losses = self._forward_fusion_losses(
            content_image_embeds,
            content_query_outputs,
            style_query_outputs,
            batch['label'],
        )
        total_loss = sum(
            [
                self.loss_weights[0] * fusion_losses['cls_loss'],
                self.loss_weights[1] * content_lm_loss,
                self.loss_weights[2] * style_lm_loss,
            ]
        )

        return {
            'loss': total_loss,
            'cls_loss': fusion_losses['cls_loss'],
            'content_lm_loss': content_lm_loss,
            'style_lm_loss': style_lm_loss,
        }

    @ torch.no_grad()
    def generate(self, batch):
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
        return {
            'scores': fusion_preds['cls_preds'],
        }

    def forward_train(self, batch):
        return self(batch)

    def forward_test(self, batch):
        return self.generate(batch)

    @ torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
