from typing import Tuple

import clip
import numpy as np
import torch
import torch.nn as nn

from ..base import BaseModelInterface
from ..nn import InfoNCELoss
from .text_template import REAL_TEMPLATEs, SPOOF_TEMPLATEs


class FLIPMCL(BaseModelInterface):
    def __init__(
        self,
        input_size: Tuple[int, int] = (224, 224),
        ssl_mlp_dim: int = 256,
        ssl_emb_dim: int = 256,
        modality: str = 'rgb',
    ):
        super().__init__()
        # load the model
        self.backbone, _ = clip.load("ViT-B/16", device="cpu")
        x = torch.randn(1, 3, *input_size)
        feat_dim = self.backbone.encode_image(x).shape[-1]

        # define the SSL parameters
        self.image_mlp = nn.Sequential(
            nn.Linear(feat_dim, ssl_mlp_dim),
            nn.BatchNorm1d(ssl_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ssl_mlp_dim, ssl_mlp_dim),
            nn.BatchNorm1d(ssl_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ssl_mlp_dim, ssl_emb_dim),
        )

        # tokenize the spoof and real templates
        with torch.no_grad():
            spoof_texts = clip.tokenize(SPOOF_TEMPLATEs)    # tokenize
            real_texts = clip.tokenize(REAL_TEMPLATEs)      # tokenize
            all_spoof_text_embs = self.backbone.encode_text(spoof_texts)
            all_real_text_embs = self.backbone.encode_text(real_texts)
        self.register_buffer("all_spoof_text_embs", all_spoof_text_embs)
        self.register_buffer("all_real_text_embs", all_real_text_embs)

        # stack the embeddings for image-text similarity
        with torch.no_grad():
            spoof_class_embeddings = all_spoof_text_embs.mean(dim=0)
            real_class_embeddings = all_real_text_embs.mean(dim=0)
            ensemble_weights = [spoof_class_embeddings, real_class_embeddings]
            text_features = torch.stack(ensemble_weights, dim=0)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_features_norm", text_features_norm)

        self.modality = modality
        if modality != 'rgb':
            raise NotImplementedError(f"Unsupported modality: {modality}")

        self.ce_loss = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineSimilarity()
        self.mse_loss = nn.MSELoss()
        self.info_nce_loss = InfoNCELoss(n_views=2, temperature=0.1)

    def _ssl_text_similarity(self, ssl_feats1, ssl_feats2, source_labels, ):
        text_embed_v1 = []
        text_embed_v2 = []
        all_spoof_text_embs = self.all_spoof_text_embs
        all_real_text_embs = self.all_real_text_embs

        for label in source_labels:
            label = int(label.item())
            if label == 0:  # spoof
                # Randomly choose indices for the 2 views
                available_indices = np.arange(0, len(all_spoof_text_embs))
                pair_1 = np.random.choice(available_indices, len(all_spoof_text_embs)//2)
                pair_2 = np.setdiff1d(available_indices, pair_1)
                # slice embedding by the indices based on the tokenized fake templates
                spoof_texts_v1 = [all_spoof_text_embs[i] for i in pair_1]
                spoof_texts_v2 = [all_spoof_text_embs[i] for i in pair_2]
                # stack the embeddings
                spoof_texts_v1 = torch.stack(spoof_texts_v1)  # 3x512
                spoof_texts_v2 = torch.stack(spoof_texts_v2)  # 3x512
                # append the embeddings
                text_embed_v1.append(spoof_texts_v1.mean(dim=0))
                text_embed_v2.append(spoof_texts_v2.mean(dim=0))
            elif label == 1:  # real
                # Randomly choose indices for the 2 views
                available_indices = np.arange(0, len(all_real_text_embs))
                pair_1 = np.random.choice(available_indices, len(all_real_text_embs)//2)
                pair_2 = np.setdiff1d(available_indices, pair_1)
                # slice embedding by the indices based on the tokenized real templates
                real_texts_v1 = [all_real_text_embs[i] for i in pair_1]
                real_texts_v2 = [all_real_text_embs[i] for i in pair_2]
                # stack the embeddings
                real_texts_v1 = torch.stack(real_texts_v1)  # 3x512
                real_texts_v2 = torch.stack(real_texts_v2)  # 3x512
                # append the embeddings
                text_embed_v1.append(real_texts_v1.mean(dim=0))
                text_embed_v2.append(real_texts_v2.mean(dim=0))

        text_embed_v1 = torch.stack(text_embed_v1)          # Bx512
        text_embed_v2 = torch.stack(text_embed_v2)          # Bx512

        # dot product of image and text embeddings
        ssl_feats1_norm = ssl_feats1 / ssl_feats1.norm(dim=-1, keepdim=True)
        ssl_feats2_norm = ssl_feats2 / ssl_feats2.norm(dim=-1, keepdim=True)

        text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim=-1, keepdim=True)
        text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim=-1, keepdim=True)

        ss1_text_dot_product = self.cosine_similarity(ssl_feats1_norm, text_embed_v1_norm)
        ss2_text_dot_product = self.cosine_similarity(ssl_feats2_norm, text_embed_v2_norm)

        # mse loss between the dot product of aug1 and aug2
        dot_product_loss = self.mse_loss(ss1_text_dot_product, ss2_text_dot_product)
        return dot_product_loss

    def _get_image_text_similarity(self, xs):
        # get the norm vector of image features
        image_features = self.backbone.encode_image(xs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.backbone.logit_scale.exp()
        sim_matrix = logit_scale * image_features @ self.text_features_norm.t()
        return sim_matrix

    def forward(
        self,
        xs: torch.Tensor,
        is_train: bool = False,
        ssl_img1: torch.Tensor = None,
        ssl_img2: torch.Tensor = None,
    ):
        sim_matrix = self._get_image_text_similarity(xs)
        if is_train:
            ssl_feats1 = self.backbone.encode_image(ssl_img1)
            ssl_feats2 = self.backbone.encode_image(ssl_img2)
            ssl_feats1_embed = self.image_mlp(ssl_feats1)
            ssl_feats2_embed = self.image_mlp(ssl_feats2)
            return sim_matrix, ssl_feats1, ssl_feats2, ssl_feats1_embed, ssl_feats2_embed
        else:
            return sim_matrix

    def forward_train(self, batch: dict):
        img = batch['img']
        ssl_img1 = batch['ssl_img1']
        ssl_img2 = batch['ssl_img2']
        labels = batch['label']
        sim_matrix, ssl_feats1, ssl_feats2, ssl_feats1_emb, ssl_feats2_emb = self(
            xs=img,
            is_train=True,
            ssl_img1=ssl_img1,
            ssl_img2=ssl_img2,
        )
        logits_ssl, labels_ssl = self.info_nce_loss(ssl_feats1_emb, ssl_feats2_emb)
        dot_product_loss = self._ssl_text_similarity(ssl_feats1, ssl_feats2, labels)
        cls_loss = self.ce_loss(sim_matrix, labels)
        sim_loss = self.ce_loss(logits_ssl, labels_ssl)
        loss = cls_loss + sim_loss + dot_product_loss
        return {'loss': loss}

    @torch.no_grad()
    def forward_test(self, batch: dict):
        logits = self(batch['img'], is_train=False)
        scores = torch.softmax(logits, dim=-1)[..., 1]
        return {
            'scores': scores,
            'logits': logits,
        }

    @torch.no_grad()
    def show_detail(self, *args, **kwargs):
        return
