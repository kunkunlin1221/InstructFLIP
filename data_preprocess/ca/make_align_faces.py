from pathlib import Path

import fire
import mmcv
import numpy as np
from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
from tqdm import tqdm


def crop_face(image, face_detector, enable_wilder=False):
    if image.shape[0] > 1280:
        image = mmcv.imrescale(image, 1280 / image.shape[0])
    results = face_detector.detect_face(image)
    if results is None:
        image = mmcv.impad(image, shape=(max(image.shape), max(image.shape)))
        results = face_detector.detect_face(image)
    if results is not None:
        points = results[1]
        ratio = 0.37 if not enable_wilder else 0.74
        cropped_img = face_detector.extract_image_chips(image, points, 224, ratio)[0]
        return cropped_img


def main(data_folder, out_folder, enable_wilder=False):
    data_folder = Path(data_folder)
    out_folder = Path(out_folder)
    img_fpaths = [x for x in data_folder.rglob('Data/**/*.*') if x.suffix in ['.jpg', '.png']]
    face_detector = MtcnnDetector(
        num_worker=4,
        accurate_landmark=True,
        threshold=[0.5, 0.5, 0.5],
    )

    for img_fpath in tqdm(img_fpaths, desc='Processing images'):
        dst_fpath = out_folder / img_fpath.relative_to(data_folder).with_suffix('.jpg')
        if not dst_fpath.exists():
            image = mmcv.imread(img_fpath)
            cropped_img = crop_face(image, face_detector, enable_wilder)

            if cropped_img is None:
                ann_file = img_fpath.parent / (img_fpath.stem + '_BB.txt')
                with open(ann_file) as f:
                    x, y, w, h, s = [float(x) for x in f.read().strip().split(" ")]
                real_h, real_w = image.shape[:2]
                x = int(real_w * x / 224)
                y = int(real_h * y / 224)
                w = int(real_w * w / 224)
                h = int(real_h * h / 224)
                bboxes = np.array([x, y, x + w, y + h])
                cx, cy = (bboxes[:2] + bboxes[2:]) // 2
                if enable_wilder:
                    w *= 2
                    h *= 2
                bboxes = np.array([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2])
                cropped_img = mmcv.imcrop(image, bboxes)
                cropped_img = mmcv.imresize(cropped_img, (224, 224))
                mmcv.imwrite(cropped_img, dst_fpath, auto_mkdir=True)
                mmcv.imwrite(cropped_img, 'test.jpg')

            mmcv.imwrite(cropped_img, dst_fpath, auto_mkdir=True)


if __name__ == '__main__':
    fire.Fire(main)
