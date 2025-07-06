from pathlib import Path

import mmcv
import torch
from fire import Fire
from PIL import Image

from src.models import InstructFLIPT5V2


def main(device='cuda'):
    content_prompt = (
        "Which type of persentation attack is in this image?\n"
        "(1) real face (2) printed photo (3) replay (4) 2D mask (5) None of the above\n"
    )
    model = InstructFLIPT5V2().to(device=device)
    model.lazy_init(0)
    model = model.train()
    samples = {
        'img': torch.randn((1, 3, 224, 224), device=device),
        'text_input': [content_prompt],
        'text_output': ['(1) real face'],
        'label': torch.zeros((1, ), dtype=torch.long, device='cuda'),
    }
    losses = model.forward_train(samples)
    with torch.no_grad():
        model = model.eval()
        outputs = model.forward_test(samples)
    breakpoint()


Fire(main)
