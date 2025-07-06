from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from fire import Fire
from PIL import Image
from sklearn.utils import resample




def train(data_folder):
    pretrained_name = "Salesforce/instructblip-flan-t5-xl"
    files = list(Path(data_folder).rglob('**/*.jpg'))
    model = InstructBlipForConditionalGeneration.from_pretrained(
        pretrained_name,
        device_map='cuda',
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
    )
    freeze_model(model.vision_model)
    freeze_model(model.language_model)
    freeze_model(model.language_projection)
    model = model.train()

    processor = InstructBlipProcessor.from_pretrained(pretrained_name)
    content_prompt = (
        "Which type of persentation attack is in this image?\n"
        "(1) real face (2) printed photo (3) replay (4) 2D mask (5) None of the above\n"
    )
    # content_prompt = (
    #     "What is unusual about this image?"
    # )
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
    style_ans_pool = [
        ' (1) indoor',
        ' (1) normal',
    ]
    max_length = max(len(prompt) for prompt in style_prompt_pool)
    max_src_length = max(max_length, len(content_prompt))
    max_tgt_length = 16

    generate_kwargs = {
        # "do_sample": False,
        # "num_beams": 5,
        # "max_length": max_tgt_length,
        # "min_length": 1,
        # "top_p": 0.9,
        # "repetition_penalty": 1.5,
        # "length_penalty": 1.0,
        # "temperature": 1,
    }
    batch_size = 1
    for file in files:
        print(file)
        model = model.train()
        image = Image.open(file).convert("RGB")
        images = [image] * batch_size
        content_questions = [content_prompt] * batch_size
        style_inds = resample(range(len(style_prompt_pool)), n_samples=batch_size)
        style_questions = [style_prompt_pool[ind] for ind in style_inds]
        content_inputs = processor(
            images=images,
            text=content_questions,
            return_tensors='pt',
            padding="max_length",
            max_length=max_src_length,
            truncation=True,
        ).to(device='cuda')
        style_inputs = processor(
            text=style_questions,
            padding="max_length",
            return_tensors='pt',
            max_length=max_src_length,
            truncation=True,
        ).to(device='cuda')
        content_inputs['pixel_values'] = content_inputs['pixel_values'].to(dtype=torch.bfloat16)
        style_inputs['pixel_values'] = content_inputs['pixel_values']

        content_ans = ['(1) real face'] * batch_size
        style_ans = [style_ans_pool[ind] for ind in style_inds]

        content_labels = processor(
            text=content_ans,
            return_tensors='pt',
            padding="max_length",
            max_length=max_tgt_length,
            truncation=True,
        ).input_ids
        style_labels = processor(
            text=style_ans,
            return_tensors='pt',
            padding="max_length",
            max_length=max_tgt_length,
            truncation=True,
        ).input_ids

        # content_pads = torch.full((batch_size, 1), processor.tokenizer.pad_token_id).type_as(content_labels)
        # style_pads = torch.full((batch_size, 1), processor.tokenizer.pad_token_id).type_as(style_labels)
        # content_labels = torch.cat([content_pads, content_labels[..., :-1]], dim=1)
        # style_labels = torch.cat([style_pads, style_labels[..., :-1]], dim=1)

        # content_labels[content_labels == processor.tokenizer.pad_token_id] = -100
        # style_labels[style_labels == processor.tokenizer.pad_token_id] = -100

        train_content_outputs = model(**content_inputs, labels=content_labels)
        train_style_outputs = model(**style_inputs, labels=style_labels)

        model = model.eval()
        with torch.no_grad():
            generate_content_outputs = model.generate(
                **content_inputs,
                **generate_kwargs,
            )
            generate_style_outputs = model.generate(
                **style_inputs,
                **generate_kwargs,
            )
            # generate_content_outputs = [
            #     x.strip() for x in processor.batch_decode(generate_content_outputs, skip_special_tokens=False)
            # ]
            # generate_style_outputs = [
            #     x.strip() for x in processor.batch_decode(generate_style_outputs, skip_special_tokens=False)
            # ]
        pprint(
            {
                'content_loss': train_content_outputs.loss,
                'style_loss': train_style_outputs.loss,
                'content_qa': {
                    '1_question': content_questions[0],
                    '2_gt': content_ans[0],
                    '2_pred': generate_content_outputs[0],
                    '3_logits': train_content_outputs.logits.argmax(dim=-1),
                },
                'style_qa': {
                    '1_question': style_questions[0],
                    '2_gt': style_ans[0],
                    '2_pred': generate_style_outputs[0],
                    '3_logits': train_style_outputs.logits.argmax(dim=-1),
                }
            }
        )
        breakpoint()


Fire(train)
