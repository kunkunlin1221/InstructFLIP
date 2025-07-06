import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseModelInterface
from ..nn.utils import freeze_model
from .components import ImageEncoder, QFormer, TextEncoder, TextSupervision


class CFPL(BaseModelInterface):
    text_pool = [
        "A photo of a real face",
        "A photo of a fake face",
    ]

    def __init__(
        self,
        clip_name: str = "ViT-B/16",
        num_queries: int = 16,
        ptm_queue_size: int = 4096,
    ):
        super().__init__()
        self._build_model(clip_name, num_queries, ptm_queue_size)
        self.ce_loss = nn.CrossEntropyLoss()

    def _build_model(self, clip_name, num_queries, ptm_queue_size):
        self.image_encoder = ImageEncoder(clip_name)
        self.text_encoder = TextEncoder(clip_name)
        self.text_supervision = TextSupervision(clip_name, num_queries=num_queries)

        with torch.no_grad():
            content_feat, layer_feats = self.image_encoder(torch.randn(1, 3, 224, 224))
            content_dim = content_feat.shape[-1]
            style_dim = layer_feats[0].shape[-1] * 2

        self.content_qformer = QFormer(num_queries, content_dim, content_dim)
        self.content_qformer_m = QFormer(num_queries, content_dim, content_dim)
        freeze_model(self.content_qformer_m)

        self.style_qformer = QFormer(num_queries, content_dim, style_dim)

        self.style_factor = torch.distributions.Beta(0.1, 0.1)

        self.prompt_modulation = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim // 16),
            nn.ReLU(True),
            nn.Linear(content_dim // 16, content_dim),
            nn.Sigmoid(),
        )
        self.class_head = nn.Linear(content_dim, 2)

        self.ptm_queue_size = ptm_queue_size
        text_queue = torch.randn(512, 16, ptm_queue_size)
        pc_queue = torch.randn(512, 16, ptm_queue_size)
        self.register_buffer("text_queue", F.normalize(text_queue, dim=0))
        self.register_buffer("pc_queue", F.normalize(pc_queue, dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.momentum = 0.995
        self.model_pairs = [
            [self.content_qformer, self.content_qformer_m],
        ]
        self.ptm_head = nn.Linear(content_dim * 2, 2)

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _update_queue(self, text_sups, content_prompt):
        self.text_queue = torch.cat([self.text_queue, text_sups], dim=-1)
        self.text_queue = self.text_queue[..., -self.ptm_queue_size :]
        self.pc_queue = torch.cat([self.pc_queue, content_prompt], dim=-1)
        self.pc_queue = self.pc_queue[..., -self.ptm_queue_size :]

    @torch.no_grad()
    def gen_text_supervision(self, texts: str):
        return self.text_supervision(texts)

    def forward(self, xs: torch.Tensor, texts: List[str] = None, is_train: bool = False):
        content_feat, layer_feats = self.image_encoder(xs)

        _, b, d = layer_feats[0].shape
        style_feat = torch.zeros(b, d * 2, device=layer_feats[0].device, dtype=layer_feats[0].dtype)

        for feat in layer_feats:
            feat = feat.permute(1, 0, 2)  # LBD -> BLD,  L = HW
            avg_feat, std_feat = feat.mean(dim=1), feat.std(dim=1)
            style_feat = style_feat + torch.cat([avg_feat, std_feat], dim=-1) / len(layer_feats)

        if is_train and random.randint(0, 1):
            b = style_feat.shape[0]
            factor = self.style_factor.sample_n(b)[..., None].to(device=style_feat.device)
            inds = torch.randperm(b)
            style_feat = style_feat * factor + style_feat[inds] * (1 - factor)

        pc_embeds = self.content_qformer(content_feat)
        ps_embeds = self.style_qformer(style_feat)

        content_text_embeds = self.text_encoder(pc_embeds)
        style_text_embeds = self.text_encoder(ps_embeds)
        text_embeds = torch.cat([content_text_embeds, style_text_embeds], dim=-1)
        w = self.prompt_modulation(text_embeds)
        cls_logits = self.class_head(w * content_feat)

        # L ptm
        if is_train:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
                self._momentum_update()
                pc_embeds_m = self.content_qformer_m(content_feat)
                pc_feats_m = F.normalize(pc_embeds_m, dim=-1)
                pc_feats_all = torch.cat([pc_feats_m.permute(2, 1, 0), self.pc_queue.clone().detach()], dim=-1)
                text_sups_embeds = self.gen_text_supervision(texts)
                text_sups_feats = F.normalize(text_sups_embeds, dim=-1)
                text_sups_feats_all = torch.cat(
                    [text_sups_feats.permute(2, 1, 0), self.text_queue.clone().detach()],
                    dim=-1,
                )
            pc_feats = F.normalize(pc_embeds, dim=-1)
            sim_p2t = torch.einsum("bik,kic->bc", pc_feats, text_sups_feats_all) / self.temp
            sim_t2p = torch.einsum("bik,kic->bc", text_sups_feats, pc_feats_all) / self.temp

            self._update_queue(text_sups_feats.T, pc_feats_m.T)

            # positive samples
            with torch.no_grad():
                bs = xs.size(0)
                weights_p2t = F.softmax(sim_p2t[:, :bs], dim=1)
                weights_t2p = F.softmax(sim_t2p[:, :bs], dim=1)

                weights_p2t.fill_diagonal_(0)
                weights_t2p.fill_diagonal_(0)

            # select a negative prompt for each text
            pc_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2p[b], 1).item()
                pc_embeds_neg.append(pc_embeds[neg_idx])
            pc_embeds_neg = torch.stack(pc_embeds_neg, dim=0)

            # select a negative text for each prompt
            text_sups_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_p2t[b], 1).item()
                text_sups_embeds_neg.append(text_sups_embeds[neg_idx])
            text_sups_embeds_neg = torch.stack(text_sups_embeds_neg, dim=0)

            rp = torch.cat([pc_embeds, text_sups_embeds], dim=-1)
            rnp = torch.cat([pc_embeds_neg, text_sups_embeds], dim=-1)
            rnt = torch.cat([pc_embeds, text_sups_embeds_neg], dim=-1)

            r = torch.cat([rp, rnp, rnt], dim=0)
            b, n, d = r.shape

            ptm_preds = self.ptm_head(r.view(-1, d)).view(b, n, 2).softmax(dim=-1).mean(dim=1)
            ptm_y = torch.cat([torch.ones(bs), torch.zeros(bs * 2)], dim=0)
            ptm_y = ptm_y.to(dtype=torch.long, device=ptm_preds.device)

            return cls_logits, ptm_preds, ptm_y
        else:
            return cls_logits

    def forward_train(self, batch):
        imgs = batch["img"]
        labels = batch["label"]
        texts = [self.text_pool[l] for l in labels]  # noqa: E741

        cls_preds, ptm_preds, ptm_y = self(imgs, texts, is_train=True)

        # binary loss
        loss_ce = self.ce_loss(cls_preds, labels)
        loss_ptm = self.ce_loss(ptm_preds, ptm_y)
        loss = loss_ce + loss_ptm
        return {"loss": loss}

    @torch.no_grad()
    def forward_test(self, batch: dict):
        imgs = batch["img"]
        logits = self(imgs, None, is_train=False)
        scores = torch.softmax(logits, dim=-1)[..., 1]
        return {
            "scores": scores,
            "logits": logits,
        }

    @torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
