import mmcv
import numpy as np
from fire import Fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.rgb.rgb_dataset import BalanceRGBDataset


def main():
    kwargs_list = [
        {
            'data_folder': '/data/disk2/fas/processed/rgb/aligned/MCIO/msu',
            'train': True,
            'mode': 'live_only',
        },
        {
            'data_folder': '/data/disk2/fas/processed/rgb/aligned/MCIO/msu',
            'train': True,
            'mode': 'fake_only',
        },
        {
            'data_folder': '/data/disk2/fas/processed/rgb/aligned/MCIO/oulu',
            'train': True,
            'mode': 'live_only',
        },
        {
            'data_folder': '/data/disk2/fas/processed/rgb/aligned/MCIO/oulu',
            'train': True,
            'mode': 'fake_only',
        },
    ]
    dataset = BalanceRGBDataset(kwargs_list, dataset_size=20000)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=16, shuffle=False)
    for _, batch in enumerate(tqdm(dataloader)):
        restore_imgs = dataset.restore_batch_imgs(batch['img'])
        plotted = np.concatenate(restore_imgs, axis=1)
        plotted = mmcv.rgb2bgr(plotted) if dataset.color_channel == 'rgb' else plotted
        mmcv.imwrite(plotted, 'rgb_dataset.png')
    print(len(dataset))


if __name__ == "__main__":
    Fire(main)
