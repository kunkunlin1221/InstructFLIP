import math
from pathlib import Path
from shutil import copy

import mmcv
from fire import Fire
from tqdm import tqdm

dataset_mapping = {
    'casia': 'CASIA_FASD',
    'msu': 'MSU_MFSD',
    'oulu': 'oulu',
    'replay': 'replayattack',
    'cefa': 'CASIA_CeFA_Race',
    'surf': 'CASIA_SURF_Challenge',
    'wmca': 'WMCA',
    'celeb': 'CelebA_Spoof',
}


def get_casia_fasd_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    stem = dst_path.stem
    x1, *x2, _ = stem.split('_')
    stem = '_'.join(x2)
    if dst_path.parts[1] == 'train':
        x0 = 'train_release'
    else:
        x0 = 'test_release'
    path = Path(data_dir, dataset, x0, x1, f'{stem}.avi')
    return path


def get_msu_mfsd_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    stem = dst_path.stem.split('_frame0')[0]
    if dst_path.parts[2] == 'fake':
        x0 = 'attack'
    else:
        x0 = 'real'
    path1 = Path(data_dir, dataset, 'scene01', x0, f'{stem}.mp4')
    path2 = Path(data_dir, dataset, 'scene01', x0, f'{stem}.mov')
    if path1.exists():
        path = path1
    elif path2.exists():
        path = path2
    return path


def get_oulu_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    if dst_path.parts[1] == 'train':
        x0 = 'Train_files'
    else:
        x0 = 'Test_files'
    stem = dst_path.stem.split('_frame0')[0]
    path = Path(data_dir, dataset, x0, f"{stem}.avi")
    return path


def get_replayattack_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    if dst_path.parts[1] == 'train':
        x0 = 'train'
    else:
        x0 = 'test'
    if dst_path.parts[2] == 'fake':
        x1 = 'attack'
        x2, *x3, _ = dst_path.stem.split('_')
        stem = '_'.join(x3)
        path = Path(data_dir, dataset, x0, x1, x2, f"{stem}.mov")
    elif dst_path.parts[2] == 'real':
        x1 = 'real'
        x2, *x3, _ = dst_path.stem.split('_')
        stem = '_'.join(x3)
        path = Path(data_dir, dataset, x0, x1, f"{stem}.mov")
    return path


def get_celeba_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    x0, x1, x2 = dst_path.stem.split('_')
    path = Path(data_dir, dataset, 'Data', 'train', x0, x1, f"{x2}.jpg")
    return path


def get_casia_surf_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    mode = dst_path.parts[1]
    if mode == 'train':
        parts = dst_path.stem.split('_')
        if parts[0] == 'Testing':
            x0, x1, x2 = parts
            path = Path(data_dir, dataset, x0, x1, f"{x2}.jpg")
        elif parts[1] == 'real':
            x0 = parts[0]
            x1 = '_'.join(parts[1:3])
            x2 = '_'.join(parts[3:5])
            x3 = parts[5]
            x4 = parts[6]
            x5 = parts[7]
            path = Path(data_dir, dataset, x0, x1, x2, x3, x4, f"{x5}.jpg")
        else:
            x0 = parts[0]
            x1 = '_'.join(parts[1:3])
            x2 = '_'.join(parts[3:5])
            x3 = '_'.join(parts[5:8])
            x4 = parts[8]
            x5 = parts[9]
            path = Path(data_dir, dataset, x0, x1, x2, x3, x4, f"{x5}.jpg")
    else:
        parts = dst_path.stem.split('_')
        x0 = parts[0]
        x1 = parts[1]
        x2 = parts[2]
        path = Path(data_dir, dataset, x0, x1, x2 + '.jpg')
    return path


def get_casia_cefa_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    *sub_folder, n_frame = dst_path.stem.split('_')
    if sub_folder[0] == '1':
        f1 = 'AF'
        f2 = f"AF-{sub_folder[1]}"
    elif sub_folder[0] == '2':
        f1 = 'CA'
        f2 = f"CA-{sub_folder[1]}"
    elif sub_folder[0] == '3':
        f1 = 'EA'
        f2 = f"EA-{sub_folder[1]}"
    f3 = "_".join(sub_folder)
    path = Path(data_dir, dataset, f1, f2, f3, 'profile', f"{int(n_frame)+1:04d}.jpg")
    return path


def get_wmca_path(data_dir, dst_path):
    dataset = dataset_mapping[dst_path.parts[0]]
    x0, *x1, nframe = dst_path.stem.split('_')
    folder_name = '_'.join(x1)
    # run wmca.py first
    path = Path(data_dir, dataset, '_extracted', 'data', folder_name, 'color', f'{int(nframe):04d}.jpg')
    return path


def get_raw_path(data_dir, dst_path):
    name = dataset_mapping[dst_path.parts[0]]
    path = None
    if name == 'CASIA_FASD':
        path = get_casia_fasd_path(data_dir, dst_path)
    elif name == 'MSU_MFSD':
        path = get_msu_mfsd_path(data_dir, dst_path)
    elif name == 'oulu':
        path = get_oulu_path(data_dir, dst_path)
    elif name == 'replayattack':
        path = get_replayattack_path(data_dir, dst_path)
    elif name == 'CelebA_Spoof':
        path = get_celeba_path(data_dir, dst_path)
    elif name == 'WMCA':
        path = get_wmca_path(data_dir, dst_path)
    elif name == 'CASIA_CeFA_Race':
        path = get_casia_cefa_path(data_dir, dst_path)
    elif name == 'CASIA_SURF_Challenge':
        path = get_casia_surf_path(data_dir, dst_path)
    return path


def main(
    data_txt_dir: str = 'data_preprocess/rgb/MCIO',
    raw_data_dir: str = 'data/data',
    dst_dir: str = 'data/processed/rgb/_extracted/MCIO',
):
    data_txt_dir = Path(data_txt_dir)
    dst_dir = Path(dst_dir)

    txt_files = sorted(data_txt_dir.glob('*.txt'))

    for txt_file in txt_files:
        with open(txt_file) as f:
            files = f.readlines()
            files = [Path(x.strip()) for x in files]
        for file in tqdm(files, desc=txt_file.stem):
            raw_path = get_raw_path(raw_data_dir, file)
            is_video = raw_path.suffix in ['.avi', '.mp4', '.mov']

            if is_video:
                video_reader = mmcv.VideoReader(str(raw_path))

                frame0_fpath = Path(dst_dir, file)
                if not frame0_fpath.exists():
                    frame0 = video_reader[6]
                    mmcv.imwrite(frame0, frame0_fpath, auto_mkdir=True)

                frame1_fpath = Path(str(frame0_fpath).replace('frame0', 'frame1'))
                if not frame1_fpath.exists():
                    frame1 = video_reader[6 + math.floor(video_reader.frame_cnt/2)]
                    mmcv.imwrite(frame1, frame1_fpath, auto_mkdir=True)

            else:
                frame_fpath = Path(dst_dir, file)
                if not frame_fpath.exists():
                    frame_fpath.parent.mkdir(parents=True, exist_ok=True)
                    copy(raw_path, frame_fpath)


if __name__ == '__main__':
    Fire(main)
