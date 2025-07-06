from pprint import pprint

import torch
from fire import Fire

from src.dataset import CAInstructDataset
from src.models import build_model
from src.utils.tools import load_yaml


def load_model(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cuda')['state_dict']
    tmp_state_dict = {k[6:]: v for k, v in state_dict.items()}
    model.load_state_dict(tmp_state_dict)
    del state_dict
    torch.cuda.empty_cache()
    return model


def main(csv_file, model_cfg, ckpt_path):
    device = 'cuda'
    dtype = torch.bfloat16
    model_cfg = load_yaml(model_cfg)
    model = build_model(**model_cfg['lm']['model']).to(device=device, dtype=dtype)
    model.lazy_init(dtype=dtype)
    model = load_model(model, ckpt_path).eval()
    dataset = CAInstructDataset(csv_file=csv_file)

    for data in dataset:
        xs = data['img'].unsqueeze(0).to(device=device, dtype=dtype)
        outputs = {'label': 'fake' if data['label'] else 'live'}
        num_styles = len(data['style_inputs'])
        batch = {
            'img': torch.cat([xs] * num_styles, dim=0),
            'content_input': [data['content_input']] * num_styles,
            'style_input': data['style_inputs'],
        }
        outputs = model.generate(batch)
        shows = {
            'label': data['label'],
            'content': {
                'insturction': data['content_input'],
                'response': outputs['content_output_texts'][0],
                '_ground_truth': data['content_output'],
            },
        }
        shows.update(
            {
                f'style_{i}': {
                    'insturction': data['style_inputs'][i],
                    'response': outputs['style_output_texts'][i],
                    '_ground_truth': data['style_outputs'][i],
                }
                for i in range(num_styles)
            }
        )
        pprint(shows)
        input("Press enter to continue...")


Fire(main)
