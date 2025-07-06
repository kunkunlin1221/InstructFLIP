import mmcv
import numpy as np
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CAInstructDatasetV2


def main(csv_file: str):
    dataset = CAInstructDatasetV2(csv_file, train=True, ssl_mode=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for _, batch in enumerate(tqdm(dataloader)):
        breakpoint()
    print(len(dataset))


if __name__ == "__main__":
    Fire(main)
