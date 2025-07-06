from pprint import pprint

from fire import Fire
from lavis.models import load_model_and_preprocess
from PIL import Image

from src.dataset.instruct._ca_instruction import (camera_quality_questions,
                                                  environment_questions,
                                                  illumination_questions,
                                                  spoof_type_questions)


def main(img_file):
    model, vis_process, _ = load_model_and_preprocess(
        name="blip2_t5_instruct",
        model_type="flant5xl",
        is_eval=True,
    )
    img = Image.open(img_file)
    img = vis_process['eval'](img).unsqueeze(0)
    content = model.generate({"image": img, "prompt": spoof_type_questions})
    style1 = model.generate({"image": img, "prompt": illumination_questions})
    style2 = model.generate({"image": img, "prompt": environment_questions})
    style3 = model.generate({"image": img, "prompt": camera_quality_questions})
    pprint(
        {
            "content": content,
            "style1": style1,
            "style2": style2,
            "style3": style3,
        }
    )
    input("Press enter to continue...")


Fire(main)
