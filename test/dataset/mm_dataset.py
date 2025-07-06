from random import shuffle

import mmcv
from fire import Fire
from tqdm import tqdm

from src.dataset.mm_dataset import MMDataset


def main(label_file: str):
    dataset = MMDataset(label_file, (224, 224))

    inds = list(range(len(dataset)))
    shuffle(inds)
    for i in tqdm(inds):
        data = dataset[i]
        plotted = dataset.draw_data(data)
        plotted = mmcv.rgb2bgr(plotted)
        mmcv.imwrite(plotted, "test.png")
        breakpoint()


Fire(main)
