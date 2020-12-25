import sys
import random

# OpenCV
import cv2

# Albumenatations
import albumentations.augmentations.functional as F
from albumentations.core.transforms_interface import DualTransform

sys.path.append('/workspaces/SchoolOfFishDetection_Pytorch/develop/')
import utils.transforms.functional_alb as MF


class _BaseRandomSizedCrop(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(
            self,
            h_start=0.0,
            w_start=0.0,
            h_crop_size=0,
            w_crop_size=0,
            interpolation=cv2.INTER_LINEAR,
            always_apply=False,
            p=1.0):
        super(_BaseRandomSizedCrop, self).__init__(always_apply, p)
        self.h_start = h_start
        self.w_start = w_start
        self.h_crop_size = h_crop_size
        self.w_crop_size = w_crop_size
        self.interpolation = interpolation

    def apply(
            self,
            img,
            h_start=0,
            w_start=0,
            h_crop_size=0,
            w_crop_size=0,
            interpolation=cv2.INTER_LINEAR,
            **params):
        h, w, _ = img.shape
        crop_height = min(h_crop_size, h)
        crop_width = min(w_crop_size, w)
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        return crop.copy()

    def apply_to_bbox(
            self,
            bbox,
            h_start=0,
            w_start=0,
            h_crop_size=0,
            w_crop_size=0,
            rows=0,
            cols=0,
            **params):
        crop_height = min(h_crop_size, rows)
        crop_width = min(w_crop_size, cols)
        return F.bbox_random_crop(
            bbox,
            crop_height,
            crop_width,
            h_start,
            w_start,
            rows,
            cols)

    def get_params(self):
        return {
            "h_start": self.h_start,
            "w_start": self.w_start,
            "h_crop_size": self.h_crop_size,
            "w_crop_size": self.w_crop_size,
        }


class CustomRandomSizedCrop(_BaseRandomSizedCrop):
    def __init__(
            self,
            min_size,
            max_size,
            interpolation=cv2.INTER_LINEAR,
            always_apply=False,
            p=1.0):
        super(CustomRandomSizedCrop, self).__init__(always_apply, p)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def get_params(self):
        h_crop_size = random.randint(
            self.min_size,
            self.max_size)
        w_crop_size = random.randint(
            self.min_size,
            self.max_size)
        return {
            "h_start": random.random(),
            "w_start": random.random(),
            "h_crop_size": h_crop_size,
            "w_crop_size": w_crop_size,
        }

    def get_transform_init_args_names(self):
        return "min_size", "max_size", "interpolation"


class CustomRandomResize(DualTransform):
    """Resize the input to the given height and width.
    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
            self,
            sizes,
            max_size=None,
            interpolation=cv2.INTER_LINEAR,
            always_apply=False,
            p=1):
        super(CustomRandomResize, self).__init__(always_apply, p)
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.interpolation = interpolation

    def apply(
            self,
            img,
            select_size,
            max_size,
            interpolation=cv2.INTER_LINEAR,
            **params):
        h, w, _ = img.shape
        height, weight = MF.get_size((w, h), select_size, max_size)
        return F.resize(
            img,
            height=height,
            width=weight,
            interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def get_params(self):
        size = random.choice(self.sizes)
        return {
            "select_size": size,
            "max_size": self.max_size,
        }

    def get_transform_init_args_names(self):
        return ("select_size", "max_size", "interpolation")
