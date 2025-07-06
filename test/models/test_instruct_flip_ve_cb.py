import torch
from fire import Fire

from src.models import InstructFLIP_VE_CB


def main(device='cuda'):
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
    model = InstructFLIP_VE_CB().to(device=device)
    model.lazy_init(0)
    model = model.train()
    samples = {
        'img': torch.randn((1, 3, 224, 224), device=device),
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
