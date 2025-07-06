from pathlib import Path

import mmcv
import torch
from fire import Fire
from PIL import Image

from src.models import InstructFLIPT5QLoRA


def main(data_folder, device='cuda'):
    files = list(Path(data_folder).rglob('**/*.jpg'))
    content_prompt = (
        "Which type of persentation attack is in this image?\n"
        "(1) real face (2) printed photo (3) replay (4) 2D mask (5) None of the above\n"
    )
    style_prompt_pool = [
        (
            # "Choose the correct answer to the following question: "
            "Which type of enviroment is in this image?\n"
            "(1) indoor (2) outdoor (3) None of the above\n"
        ),
        (
            # "Choose the correct answer to the following question: "
            "Which type of illumination condition is in this image?\n"
            "(1) normal (2) back (3) dark (4) None of the above\n"
        ),
    ]
    model = InstructFLIPT5QLoRA().cuda().train()
    for file in files:
        model = model.train()
        image = Image.open(str(file)).convert("RGB")
        samples = {
            'image': model.preprocess['train'](image).unsqueeze(0).to(device),
            'content_input': [content_prompt],
            'content_output': ['(1) real face'],
            'style_input': [style_prompt_pool[0]],
            'style_output': ['(1) indoor'],
            'label': torch.zeros((1, ), dtype=torch.long, device='cuda'),
        }
        losses = model.forward_train(samples)
        with torch.no_grad():
            model = model.eval()
            outputs = model.forward_test(samples)
        breakpoint()


Fire(main)
