from pathlib import Path
from pprint import pprint

import mmcv
import torch
from fire import Fire

from models.instruct_flip import InstructFLIPT5V2


def train(data_folder):
    files = list(Path(data_folder).rglob('**/*.jpg'))
    instruct_flip = InstructFLIPT5V2().train().cuda()
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    batch_size = 8
    for file in files:
        print(file)
        image = mmcv.imread(file)
        # image = image / 255
        # image = (image - mean) / std
        images = torch.from_numpy(image).permute(2, 0, 1).repeat(batch_size, 1, 1, 1).float().cuda()
        content_options = [('persentation attack', 'real')] * batch_size
        style_options = [('illumination condition', 'real')] * batch_size
        content_ans = ['real'] * batch_size
        style_ans = ['real'] * batch_size
        outputs = instruct_flip(images, content_options, style_options, content_ans, style_ans, is_train=True)
        print('content_loss:', outputs['content_lm_outputs']['loss'].item())
        print('style_loss', outputs['style_lm_outputs']['loss'].item())
        instruct_flip.eval()
        with torch.no_grad():
            outputs = instruct_flip(images, content_options, style_options, is_train=False)
        pprint(outputs)
        input("Press Enter to continue...")


Fire(train)
