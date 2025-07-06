import json
from pathlib import Path

import pandas as pd
from fire import Fire


def get_celeba_path(path):
    dataset = 'celeb'
    x0, x1, x2 = path.stem.split('_')
    path = str(Path('Data', 'train', x0, x1, f"{x2}.jpg"))
    return path


def main(
    txt_path: str = 'data_preprocess/rgb/MCIO/celeb_fake_train.txt',
    dst_csv: str = 'data_preprocess/rgb/MCIO/celeb_fake_train.csv',
    meta_folder: str = 'data/processed/ca/metas',
):
    txt_path = Path(txt_path)

    meta_files = [str(x) for x in Path(meta_folder).rglob('**/*.json') if "train" in str(x)]
    labels = {}
    for meta_file in meta_files:
        with open(meta_file) as f:
            meta = json.load(f)
        for k, v in meta.items():
            labels[k] = v[-4:] + [-1]

    ori_files = [get_celeba_path(Path(x)) for x in txt_path.read_text().splitlines()]
    files = [x for x in txt_path.read_text().splitlines()]
    out_data = {file: labels[ori_file] for file, ori_file in zip(files, ori_files)}
    columns = ['spoof_type', 'illumination', 'enviroment', 'is_fake', 'quality']
    df = pd.DataFrame.from_dict(out_data, orient='index', columns=columns)
    df.index.name = 'path'
    df.to_csv(dst_csv)


if __name__ == '__main__':
    Fire(main)
