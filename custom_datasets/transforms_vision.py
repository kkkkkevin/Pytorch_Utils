import random
import numbers

import torch
import torchvision.transforms.functional as F


def rotate(image, angle, resample, expand, center, target):
    image = F.rotate(image, angle, resample, expand, center)

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * \
            torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return image, target


def vflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [0, 3, 2, 1]] * \
            torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

    # if "masks" in target:
    #    target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)
        return img, target


class RandomRotation(object):
    def __init__(
            self,
            degrees,
            resample=False,
            expand=False,
            center=None,
            p=0.5):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.p = p

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
