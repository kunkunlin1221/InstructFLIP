import torch
from fire import Fire

from src.models.nn import CFPL


def main():
    model = CFPL("ViT-B/32")
    xs = torch.randn(8, 3, 224, 224)
    texts = ["A photo of a real face"] * 8
    cls_logits, ptm_preds, ptm_y = model(xs, texts, is_train=True)
    breakpoint()


Fire(main)
