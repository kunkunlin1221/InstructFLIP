import math
from pathlib import Path

import mmcv
from fire import Fire
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
from tqdm import tqdm


def crop_face(image, face_detector):
    if image.shape[0] > 1280:
        image = mmcv.imrescale(image, 1280 / image.shape[0])
    results = face_detector.detect_face(image)
    if results is None:
        image = mmcv.impad(image, shape=(max(image.shape), max(image.shape)))
        results = face_detector.detect_face(image)
    if results is not None:
        points = results[1]
        cropped_img = face_detector.extract_image_chips(image, points, 224, 0.37)[0]
        return cropped_img


def main(
    data_folder: str = '/data/disk2/fas/processed/rgb/_extracted',
    out_folder: str = '/data/disk2/fas/processed/rgb/aligned'
):
    data_folder = Path(data_folder)
    img_files = list(data_folder.rglob('*.*'))
    face_detector = MtcnnDetector(
        num_worker=4,
        accurate_landmark=True,
        threshold=[0.5, 0.5, 0.5],
    )

    for img_file in tqdm(img_files):
        dst_path = Path(out_folder, img_file.relative_to(data_folder))
        if not dst_path.exists():
            image = mmcv.imread(img_file)
            cropped_img = crop_face(image, face_detector)

            if cropped_img is not None:
                mmcv.imwrite(cropped_img, dst_path, auto_mkdir=True)
            else:
                print(f'Failed to process {img_file}')


if __name__ == '__main__':
    Fire(main)
