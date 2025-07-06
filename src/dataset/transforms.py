import random

from PIL import ImageFilter
from torchvision import transforms as T


class simCLRGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_train_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0),
            T.ToTensor(),
            T.Normalize(mean, std),
        ],
    )


def get_valid_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ],
    )


def get_ssl_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return T.Compose(

        [
            T.RandomResizedCrop(224, scale=(0.08, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([simCLRGaussianBlur([.1, 2.])], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
