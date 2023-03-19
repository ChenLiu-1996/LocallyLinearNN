import random
from typing import Tuple

import torchvision.transforms as transforms
from PIL import ImageFilter


class PairedAugmentation:
    '''
    Use the same position-changing augmentations, but different position-invariant augmentations.
    '''

    def __init__(self, imsize: int, mean: Tuple[float], std: Tuple[float]):
        self.pos_changing_aug = transforms.Compose([
            transforms.RandomResizedCrop(
                imsize,
                scale=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
        ])
        self.pos_invariant_aug = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        x = self.pos_changing_aug(x)
        aug1 = self.pos_invariant_aug(x)
        aug2 = self.pos_invariant_aug(x)
        return aug1, aug2


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
