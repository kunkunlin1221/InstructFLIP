from pathlib import Path
from pprint import pprint

import mmcv
import torch
from fire import Fire

from src.dataset.ca_dataset import CADataset
from models.instruct_flip._bk._instruct_flip_t5 import InstructFLIP


def train(csv_file):
    dataset = CADataset(csv_file=csv_file, train=True)
    model = InstructFLIP("Salesforce/instructblip-flan-t5-xl")
    model = model.to(dtype=torch.bfloat16, device='cuda').train()
    for data in dataset:
        breakpoint()
        print(data)
        model = model.train()
        images = data['image']
        cls_label = data['cls_label']

        content_ans = [('persentation attack', '(4) A 2D face mask')]
        style_ans = [('illumination condition', 'twilight)')]
        train_outputs = model(images, content_ans, style_ans, labels=labels, is_train=True)
        model = model.eval()
        with torch.no_grad():
            generate_outputs = model(images)
        pprint(
            {
                'content_loss': train_outputs['content_lm_loss'],
                'style_loss': train_outputs['style_lm_loss'],
                'content_generate': generate_outputs['content_lm_outputs'],
                'style_generate': generate_outputs['style_lm_outputs'],
            }
        )
        input("Press Enter to continue...")


Fire(train)
