from copy import deepcopy

import clip
import torch
import torch.nn as nn

from ..nn import CrossAttention, MultiQueryAttention, SimpleFeedForward


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ImageEncoder(nn.Module):
    def __init__(self, clip_name):
        super().__init__()
        clip_model, _ = clip.load(clip_name, jit=False, device="cpu")
        self.model = clip_model.visual

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)                                                     # [b, c, g, g]
        x = x.reshape(x.shape[0], x.shape[1], -1)                                   # [b, c, g ** 2]
        x = x.permute(0, 2, 1)                                                      # [b, g ** 2, c]
        class_embedding = self.model.class_embedding + torch.zeros_like(x[:, :1])   # [b, 1, c]
        x = torch.cat([class_embedding, x], dim=1)                                  # [b, g ** 2 + 1, c]
        x = x + self.model.positional_embedding
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # BLD -> LBD
        feats = []
        for m in self.model.transformer.resblocks:
            x = m(x)
            feats.append(x)
        x = x.permute(1, 0, 2)  # LBD -> BLD
        x = self.model.ln_post(x[:, 0, :])

        if self.model.proj is not None:
            x = x @ self.model.proj

        return x, feats


class TextEncoder(nn.Module):
    def __init__(self, clip_name):
        super().__init__()
        clip_model, _ = clip.load(clip_name, jit=False, device="cpu")
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        padding = clip_model.token_embedding(torch.zeros(1, 1, dtype=torch.long))
        self.register_buffer("padding", padding)

    def forward(self, token_embeds: torch.Tensor):
        padding = self.padding.expand(token_embeds.shape[0], 77 - token_embeds.shape[1], -1)
        token_embeds = torch.cat([token_embeds, padding], dim=1)
        x = token_embeds + self.positional_embedding
        x = x.permute(1, 0, 2)              # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)              # LND -> NLD
        x = self.ln_final(x)
        x = x[:, 16] @ self.text_projection
        return x


class TextSupervision(nn.Module):
    def __init__(self, clip_name, num_queries: int = 16):
        super().__init__()
        clip_model, _ = clip.load(clip_name, jit=False, device="cpu")
        self.token_embedding = deepcopy(clip_model.token_embedding)
        self.num_queries = num_queries

    @torch.no_grad()
    def forward(self, texts: str):
        tokenized_text = clip.tokenize(texts).to(self.token_embedding.weight.device)
        token_embeds = self.token_embedding(tokenized_text)
        token_embeds = token_embeds.mean(dim=1, keepdim=True).expand(-1, self.num_queries, -1)
        return token_embeds


class QFormerBlock(nn.Module):

    def __init__(self, query_dim: int = 16, context_n_dim: int = 512):
        super().__init__()
        self.ln = nn.LayerNorm(query_dim)
        self.msa = MultiQueryAttention(query_dim, 8, qk_ln=False)
        self.mca = CrossAttention(query_dim, context_dim=context_n_dim, heads=8, dropout=0.1, norm_context=True)
        self.ffn = SimpleFeedForward(query_dim, query_dim, dropout=0.1)

    def forward(self, queries, context_embeds: torch.Tensor):
        queries = queries + self.msa(self.ln(queries))[0]
        queries = queries + self.mca(queries, context_embeds)
        queries = queries + self.ffn(queries)
        return queries


class QFormer(nn.Module):
    def __init__(self, query_length: int = 16, query_dim: int = 512,  context_n_dim: int = 512, n_layers: int = 1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, query_length, query_dim))
        self.qformers = nn.ModuleList([QFormerBlock(query_dim, context_n_dim) for _ in range(n_layers)])

    def forward(self, context_embeds: torch.Tensor):
        b = context_embeds.shape[0]
        queries = self.queries.expand(b, -1, -1)
        context_embeds = context_embeds[:, None, :]
        for m in self.qformers:
            queries = m(queries, context_embeds)
        return queries
