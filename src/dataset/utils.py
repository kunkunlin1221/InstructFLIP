import numpy as np
import torch


class ToTensor:
    def __call__(self, xs):
        return torch.from_numpy(xs).permute(2, 0, 1).float()


class RestoreImageMixin:

    def restore_img(self, tensor):
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = (img * self.std + self.mean) * 255
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def restore_batch_imgs(self, tensor):
        imgs = tensor.cpu().numpy().transpose(0, 2, 3, 1)
        imgs = (imgs * self.std + self.mean) * 255
        imgs = np.clip(imgs, 0, 255).astype('uint8')
        return imgs
