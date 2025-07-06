from pathlib import Path

import h5py
import mmcv
import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm


def main(folder: str = '/data/disk2/fas/data/WMCA'):
    rgb_folder = Path(folder) / 'preprocessed-face-station_RGB'
    out_folder = Path(folder, '_extracted')
    out_folder.mkdir(parents=True, exist_ok=True)
    data_folder = Path(out_folder, 'data')
    rgb_files = sorted(Path(rgb_folder).rglob('*.hdf5'))

    for rgb_file in tqdm(rgb_files):
        rgbs = h5py.File(rgb_file, 'r')
        cdits = h5py.File(str(rgb_file).replace('RGB', 'CDIT'), 'r')
        dst_folder = data_folder / rgb_file.stem
        ind = 0
        if not dst_folder.exists():
            for k, rgb in rgbs.items():
                if k in cdits:
                    rgb = mmcv.rgb2bgr(np.asarray(rgb['array']).transpose(1, 2, 0))
                    cdit = np.asarray(cdits[k]['array']).transpose(1, 2, 0)
                    ind = int(k.split('_')[-1])
                    mmcv.imwrite(rgb, str(dst_folder / 'color' / f'{ind:04d}.jpg'))
                    mmcv.imwrite(cdit[..., 1], str(dst_folder / 'depth' / f'{ind:04d}.jpg'))
                    mmcv.imwrite(cdit[..., 2], str(dst_folder / 'ir' / f'{ind:04d}.jpg'))
                    mmcv.imwrite(cdit[..., 3], str(dst_folder / 'thermal' / f'{ind:04d}.jpg'))

    attack_files = pd.read_csv(Path(folder, 'documentation', 'attack_illustration_files.csv'), header=None)
    bonafide_files = pd.read_csv(Path(folder, 'documentation', 'bonafide_illustration_files.csv'), header=None)

    attack_files['is_live'] = 0
    bonafide_files['is_live'] = 1

    attack_files.iloc[:, 0] = attack_files.iloc[:, 0].apply(lambda x: Path(x).stem)
    bonafide_files.iloc[:, 0] = bonafide_files.iloc[:, 0].apply(lambda x: Path(x).stem)

    outs = pd.concat([attack_files, bonafide_files], ignore_index=True)
    outs.columns = ['folder', 'is_live']
    outs.to_csv(Path(out_folder) / 'labels.csv', index=False)


Fire(main)
