import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import clip
import matplotlib
import mmcv
import numpy as np
import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, T5TokenizerFast

from ..base import BaseModelInterface
from ..nn import CrossAttention, InfoNCELoss, MultiHeadAttention, SimpleFeedForward, init_params, normalize

os.environ["TOKENIZERS_PARALLELISM"] = "false"
matplotlib.use("Agg")


class ImageEncoder(nn.Module):
    def __init__(self, clip_name):
        super().__init__()
        clip_model, _ = clip.load(clip_name, jit=False, device="cpu")
        self.model = deepcopy(clip_model.visual)
        del clip_model

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)  # [b, c, h, w]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [b, c, h*w]
        x = x.permute(0, 2, 1)  # [b, h*w, c]
        class_embedding = self.model.class_embedding + torch.zeros_like(x[:, :1])  # [b, 1, c]
        x = torch.cat([class_embedding, x], dim=1)  # [b, 1 + h*w, c]
        x = x + self.model.positional_embedding
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)  # BLD -> LBD
        feats = []
        for m in self.model.transformer.resblocks:
            x = m(x)
            feats.append(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.model.ln_post(x)
        return x, feats


class FusionModule(nn.Module):
    def __init__(
        self,
        content_dim: int = 768,
        context_dim: int = 768,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            dim=content_dim,
            heads=heads,
            attn_impl="torch",  # 'flash' or 'triton' would get error on new pacakge
            qk_ln=True,
        )
        self.ca = CrossAttention(
            dim=content_dim,
            context_dim=context_dim,
            heads=heads,
            dropout=dropout,
            norm_context=True,
        )
        self.ffn = SimpleFeedForward(content_dim, 2048, dropout)

    def forward(self, content, queries):
        content = content + self.mha(content)[0]
        content = content + self.ca(content, queries)
        out = self.ffn(content)
        return out


class FusionModulator(nn.Module):
    def __init__(
        self,
        content_dim: int = 768,
        context_dim: int = 768,
        heads: int = 8,
        dropout: float = 0.1,
        n_layers: int = 1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([FusionModule(content_dim, context_dim, heads, dropout) for _ in range(n_layers)])
        init_params(self)

    def forward(self, content, queries):
        for _, m in enumerate(self.blocks):
            content = m(content, queries)
        return content


class Classifier(nn.Module):
    def __init__(self, n_dim, n_classes: int):
        super().__init__()
        self.layer = nn.Linear(n_dim, n_classes)
        init_params(self)

    def forward(self, x, norm_flag=True):
        if norm_flag:
            self.layer.weight.data = normalize(self.layer.weight, 2, axis=0)
        return self.layer(x)


class CueGenerator(nn.Module):
    def __init__(self, n_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(n_dim, 32, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class InstructFLIP(Blip2Base, BaseModelInterface):
    def __init__(
        self,
        loss_weights: List[float] = [0.3, 0.1, 0, 0.3, 0.3],
        t5_model_name: str = "google/flan-t5-base",
        enable_qlora_on_llm: bool = False,
        cue_noise_weight: float = 0.02,
        qformer_layers: int = 2,
        qformer_heads: int = 16,
        **kwargs,
    ):
        super().__init__()
        self._build_model(t5_model_name, qformer_layers, qformer_heads)
        self.cue_loss = nn.SmoothL1Loss(beta=0.01)
        self.ce_loss = nn.CrossEntropyLoss()
        self.info_nce_loss = InfoNCELoss(n_views=2, temperature=0.1)
        self.loss_weights = loss_weights
        self.enable_qlora_on_llm = enable_qlora_on_llm
        self.cue_noise_weight = cue_noise_weight

    @classmethod
    def init_Qformer(
        cls,
        num_query_token: int,
        vision_width: int,
        cross_attention_freq: int = 2,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 16,
    ):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.num_attention_heads = num_attention_heads
        encoder_config.layer_norm_eps = 1e-7
        Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        Qformer = Qformer.train()
        return Qformer, query_tokens

    def _build_model(self, t5_model_name, qformer_layers, qformer_heads):
        # global settings
        self.max_txt_len = 128
        self.max_output_txt_len = 16

        # vision branch
        self.visual_encoder = ImageEncoder("ViT-B/16")

        # qformer branch
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.content_Qformer, self.content_query_tokens = self.init_Qformer(
            num_query_token=32,
            vision_width=768,
            num_hidden_layers=qformer_layers,
            num_attention_heads=qformer_heads,
        )
        self.style_Qformer, self.style_query_tokens = self.init_Qformer(
            num_query_token=32,
            vision_width=768,
            num_hidden_layers=qformer_layers,
            num_attention_heads=qformer_heads,
        )
        self.content_Qformer.resize_token_embeddings(len(self.tokenizer))
        self.style_Qformer.resize_token_embeddings(len(self.tokenizer))
        self.content_Qformer.cls = None
        self.style_Qformer.cls = None

        # llm branch
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name, truncation_side="left")
        self.t5_output_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name, truncation_side="right")
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
        t5_config = T5Config.from_pretrained(self.t5_model_name)
        t5_config.dense_act_fn = "gelu"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",
        )
        t5_model = T5ForConditionalGeneration.from_pretrained(
            self.t5_model_name,
            config=t5_config,
            quantization_config=bnb_config,
            torch_dtype=dtype,
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            content_embeds, layer_feats = self.visual_encoder(image)
            style_embeds = []
            for feat in layer_feats:
                avg_feat, std_feat = feat.mean(dim=1), feat.std(dim=1)
                style_embeds.extend([avg_feat, std_feat])
            style_embeds = torch.stack(style_embeds, dim=1)  # [b, 2*L, d]
        return content_embeds, style_embeds

    # def _forward_image_embeds(self, image) -> torch.Tensor:
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         content_embeds, layer_feats = self.visual_encoder(image)
    #         b, _, d = content_embeds.shape
    #         style_embeds = torch.zeros(b, 2, d, dtype=content_embeds.dtype, device=content_embeds.device)
    #         for feat in layer_feats:
    #             avg_feat, std_feat = feat.mean(dim=1), feat.std(dim=1)
    #             style_embeds += torch.stack([avg_feat, std_feat], dim=1) / len(layer_feats)
    #     return content_embeds, style_embeds  # [b, 2, d]

    # def _forward_image_embeds(self, image) -> torch.Tensor:
    #     with torch.autocast("cuda", dtype=torch.bfloat16):
    #         content_embeds, layer_feats = self.visual_encoder(image)
    #         layer_feats = torch.stack(layer_feats)  # [b, L, d]
    #         style_embeds = torch.mean(layer_feats, dim=0)
    #     return content_embeds, style_embeds  # [b, l, d]

    def _forward_qformer(
        self,
        image_embeds: torch.Tensor,
        text_input: str,
        Qformer: nn.Module,
        query_tokens: nn.Parameter,
    ) -> torch.Tensor:
        device = image_embeds.device
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_Qformer = self.tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output = query_output.last_hidden_state[:, : query_tokens.size(1), :]
        return query_output

    def _forward_ssl_embeds(self, ssl_images: torch.Tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds, _ = self.visual_encoder(ssl_images)
            ssl_embeds = self.ssl_branch(embeds[:, 0])
        return ssl_embeds

    def _forward_lm_loss(
        self,
        query_output: torch.Tensor,
        text_input: str,
        text_output: str,
    ) -> torch.Tensor:
        device = query_output.device

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_t5 = self.t5_proj(query_output)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=device)

            input_tokens = self.t5_tokenizer(
                text_input,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            output_tokens = self.t5_output_tokenizer(
                text_output,
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
                return_tensors="pt",
            ).to(device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )
        return outputs.loss

    @torch.no_grad()
    def _forward_lm_outputs(
        self,
        query_output: torch.Tensor,
        prompts: List[str],
    ):
        device = query_output.device

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_t5 = self.t5_proj(query_output)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=device)

            input_tokens = self.t5_tokenizer(prompts, padding="longest", return_tensors="pt").to(device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                max_length=self.max_output_txt_len,
            )
        output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def _forward_fusion_losses(
        self,
        content_embeds: torch.Tensor,
        content_queries: torch.Tensor,
        style_queries: torch.Tensor,
        cls_labels: torch.Tensor,
        ssl_embeds1: torch.Tensor = None,
        ssl_embeds2: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            queries = torch.cat([content_queries, style_queries], dim=1)
            fusion_inputs = self.fusion_modulator(content_embeds, queries)
            randoms = torch.randn_like(fusion_inputs) * self.cue_noise_weight * cls_labels[..., None, None]
            randoms[:, 0] = 0  # skip cls token
            fusion_inputs += randoms
            b, l, c = fusion_inputs.shape
            cls_inputs, cue_inputs = fusion_inputs.split([1, l - 1], dim=1)
            cls_preds = self.cls_layer(cls_inputs.squeeze(1))
            h, w = int((l - 1) ** 0.5), int((l - 1) ** 0.5)
            cue_inputs = cue_inputs.permute(0, 2, 1).reshape(b, c, h, w)
            cue_preds = self.cue_generator(cue_inputs).squeeze(1)

        cls_loss = self.ce_loss(cls_preds.float().squeeze(-1), cls_labels)
        cue_targets = cls_labels[:, None, None, None].expand_as(cue_preds)
        cue_loss = self.cue_loss(cue_preds.float(), cue_targets.float())

        outputs = {
            "cls_loss": cls_loss,
            "cue_loss": cue_loss,
            "ssl_loss": torch.zeros(1, device=cls_labels.device),
        }
        if ssl_embeds1 is not None and ssl_embeds2 is not None:
            logits_ssl, labels_ssl = self.info_nce_loss(ssl_embeds1, ssl_embeds2)
            outputs["ssl_loss"] = self.ce_loss(logits_ssl, labels_ssl)

        return outputs

    @torch.no_grad()
    def _forward_fusion_preds(
        self,
        content_embeds: torch.Tensor,
        content_queries: torch.Tensor,
        style_queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            queries = torch.cat([content_queries, style_queries], dim=1)
            fusion_inputs = self.fusion_modulator(content_embeds, queries)
            b, l, c = fusion_inputs.shape
            cls_inputs, cue_inputs = fusion_inputs.split([1, l - 1], dim=1)
            cls_preds = self.cls_layer(cls_inputs.squeeze(1)).softmax(dim=-1)[:, 1]
            h, w = int((l - 1) ** 0.5), int((l - 1) ** 0.5)
            cue_inputs = cue_inputs.permute(0, 2, 1).reshape(b, c, h, w)
            cue_preds = self.cue_generator(cue_inputs)
        return {
            "cls_preds": cls_preds,
            "cue_preds": cue_preds,
        }

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        content_image_embeds, style_image_embeds = self._forward_image_embeds(batch["img"])
        content_query_outputs = self._forward_qformer(
            content_image_embeds,
            batch["content_input"],
            self.content_Qformer,
            self.content_query_tokens,
        )
        style_query_outputs = self._forward_qformer(
            style_image_embeds,
            batch["style_input"],
            self.style_Qformer,
            self.style_query_tokens,
        )
        content_inputs = []
        content_outputs = []
        content_query_inds = []
        for i, (inp, oup) in enumerate(zip(batch["content_input"], batch["content_output"])):
            if oup != "No GT":
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
        for i, (inp, oup) in enumerate(zip(batch["style_input"], batch["style_output"])):
            if oup != "No GT":
                style_inputs.append(inp)
                style_outputs.append(oup)
                style_query_inds.append(i)
        style_lm_loss = self._forward_lm_loss(
            style_query_outputs[style_query_inds],
            style_inputs,
            style_outputs,
        )
        if "ssl_img1" in batch and "ssl_img2" in batch:
            ssl_embeds1 = self._forward_ssl_embeds(batch["ssl_img1"])
            ssl_embeds2 = self._forward_ssl_embeds(batch["ssl_img2"])
        else:
            ssl_embeds1 = None
            ssl_embeds2 = None
        fusion_losses = self._forward_fusion_losses(
            content_image_embeds,
            content_query_outputs,
            style_query_outputs,
            batch["label"],
            ssl_embeds1=ssl_embeds1,
            ssl_embeds2=ssl_embeds2,
        )
        total_loss = sum([
            self.loss_weights[0] * content_lm_loss,
            self.loss_weights[1] * style_lm_loss,
            self.loss_weights[2] * fusion_losses["cls_loss"],
            self.loss_weights[3] * fusion_losses["cue_loss"],
            self.loss_weights[4] * fusion_losses["ssl_loss"],
        ])

        return {
            "loss": total_loss,
            "cls_loss": fusion_losses["cls_loss"],
            "cue_loss": fusion_losses["cue_loss"],
            "ssl_loss": fusion_losses["ssl_loss"],
            "content_lm_loss": content_lm_loss,
            "style_lm_loss": style_lm_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        batch,
        disable_lm: bool = False,
    ):
        content_image_embeds, style_image_embeds = self._forward_image_embeds(batch["img"])
        content_query_outputs = self._forward_qformer(
            content_image_embeds,
            batch["content_input"],
            self.content_Qformer,
            self.content_query_tokens,
        )
        style_query_outputs = self._forward_qformer(
            style_image_embeds,
            batch["style_input"],
            self.style_Qformer,
            self.style_query_tokens,
        )
        fusion_preds = self._forward_fusion_preds(
            content_image_embeds,
            content_query_outputs,
            style_query_outputs,
        )
        outputs = {
            "scores": fusion_preds["cls_preds"],
            "cue_preds": fusion_preds["cue_preds"],
        }
        if not disable_lm:
            outputs["content_output_texts"] = self._forward_lm_outputs(
                content_query_outputs,
                batch["content_input"],
            )
            outputs["style_output_texts"] = self._forward_lm_outputs(
                style_query_outputs,
                batch["style_input"],
            )

        return outputs

    def forward_train(self, batch):
        return self(batch)

    def forward_test(self, batch):
        return self.generate(batch, disable_lm=True)

    @torch.no_grad()
    def show_detail(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        epoch: int,
        mode: str = "train",
        logger=None,
        gpu_id: int = 0,
        dataset_name: Optional[str] = None,
    ):
        is_training = self.training
        if is_training:
            self.eval()
        num_style_questions = len(batch["style_inputs"])

        xs = batch["img"][0]
        content_inputs = [batch["content_input"][0]] * num_style_questions
        content_outputs = [batch["content_output"][0]] * num_style_questions
        style_inputs = [x[0] for x in batch["style_inputs"]]
        style_outputs = [x[0] for x in batch["style_outputs"]]

        mini_batch = {
            "img": torch.stack(
                [xs] * len(style_inputs),
                dim=0,
            ),
            "content_input": content_inputs,
            "content_output": content_outputs,
            "style_input": style_inputs,
            "style_output": style_outputs,
        }
        outputs = self.generate(mini_batch)
        # Prepare output for plotting
        content_lm_preds = outputs["content_output_texts"]
        style_lm_preds = outputs["style_output_texts"]
        cls_label = batch["label"][0]
        means = batch["mean"][0].float().cpu().numpy()
        std = batch["std"][0].float().cpu().numpy()

        img = xs.float().cpu().numpy().transpose(1, 2, 0)
        img = ((img * std + means) * 255).clip(0, 255).astype("uint8")
        img = mmcv.rgb2bgr(img)
        cls_pred = outputs["scores"][0].float().item()
        cue_pred = outputs["cue_preds"][0].float().cpu().numpy().transpose(1, 2, 0)
        cue_pred = (cue_pred * 255).clip(0, 255).astype("uint8")
        cue_img = mmcv.imresize(cue_pred, (img.shape[1], img.shape[0]))
        if logger is not None:
            plotted_dir = Path(logger.log_dir, "images", mode)
            plotted_dir.mkdir(exist_ok=True, parents=True)
            if dataset_name is None:
                fpath = plotted_dir / f"epoch={epoch}/gpu_id={gpu_id}_batch={batch_idx}"
            else:
                fpath = plotted_dir / f"{dataset_name}/epoch={epoch}/gpu_id={gpu_id}_batch={batch_idx}"
            fpath.parent.mkdir(exist_ok=True, parents=True)
            img = np.concatenate([img, cue_img], axis=1)
            mmcv.imwrite(img, str(fpath) + ".jpg")
            lm_outputs = {
                "img_path": batch["img_path"][0],
                "content_head": {
                    "question": content_inputs[0],
                    "answer": content_outputs[0],
                    "pred": content_lm_preds[0],
                    "label": "fake" if cls_label else "real",
                },
                "binary_head": {
                    "pred": "fake" if cls_pred > 0.5 else "real",
                    "label": "fake" if cls_label else "real",
                    "score": cls_pred,
                },
            }
            lm_outputs.update({
                "style_head": [
                    {
                        "question": style_inputs[i],
                        "answer": style_outputs[i],
                        "pred": style_lm_preds[i],
                    }
                    for i in range(num_style_questions)
                ]
            })
            with open(str(fpath) + ".json", "w") as f:
                json.dump(lm_outputs, f, indent=2)

        if is_training:
            self.train()
