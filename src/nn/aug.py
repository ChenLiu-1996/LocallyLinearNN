import random
from typing import Tuple

import torchvision.transforms as transforms
from PIL import ImageFilter


class PairedAugmentation:
    '''
    Use the same position-changing augmentations, but different position-invariant augmentations.
    Special Note: color transform is separated into 2 parts:
        A more drastic one included in the position-changing augmentation, and
        A more subtle one included in the position-invariant augmentation.
        The reason is we don't want the two augmented versions to be TOO different from each other.
    '''

    def __init__(self, imsize: int, mean: Tuple[float], std: Tuple[float]):
        self.pos_changing_aug = transforms.Compose([
            transforms.RandomResizedCrop(
                imsize,
                scale=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ],
                                   p=0.8),
        ])
        self.pos_invariant_aug = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.04, contrast=0.04, saturation=0.04, hue=0.01)
            ],
                                   p=0.8),
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
