import numpy as np
import random

# CV
import cv2

# Pytorch
import torch
from torch.utils.data import Dataset

# Albumenatations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import custom_datasets.transforms_alb as MT


class WheatDataset(Dataset):
    def __init__(self, image_ids, dataframe, image_dir, transforms=None):
        self.image_ids = image_ids
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg',
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        h, w, _ = image.shape  # (h,w,c)

        # DETR takes in data in yolo format
        boxes = records[['x', 'y', 'w', 'h']].values

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # (x,y,w,h) -> (x,y,x,y)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2].clip(min=0, max=h)
        # AS pointed out by PRVI It works better if the main class is labelled
        # as zero
        labels = np.zeros(len(boxes), dtype=np.int32)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes,
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

            __, h, w = image.shape  # (c, h, w)

        # convert yolo format [cx、cy、w、h]
        # and normalized
        if len(boxes) > 0:
            boxes = np.array(boxes)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] * 0.5
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] * 0.5

            boxes /= [w, h, w, h]
        else:
            boxes = np.zeros((0, 4))

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        # target['area'] = area

        return image, target, image_id


class WheatTestDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        # records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(
            f'{self.image_dir}/{image_id}.jpg',
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_train_transforms():
    return A.Compose(
        [
            A.OneOf(
                [
                    A.HueSaturationValue(hue_shift_limit=0.2,
                                         sat_shift_limit=0.2,
                                         val_shift_limit=0.2,
                                         p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2,
                                               p=0.9)
                ],
                p=0.9
            ),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),  # ng
            A.VerticalFlip(p=0.5),  # ng
            A.Resize(height=512, width=512, p=1),  # ng
            A.Cutout(
                num_holes=8,
                max_h_size=64,
                max_w_size=64,
                fill_value=0,
                p=0.5),
            ToTensorV2(p=1.0)
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels'])
    )


def get_train_transforms_v2():
    return A.Compose(
        [
            A.OneOf(
                [
                    # 色相
                    A.HueSaturationValue(hue_shift_limit=0.2,
                                         sat_shift_limit=0.2,
                                         val_shift_limit=0.2,
                                         p=0.3),
                    # 明るさとコントラスト
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2,
                                               p=0.3),
                    # RGBの各チャンネル
                    A.RGBShift(r_shift_limit=20 / 255,
                               g_shift_limit=20 / 255,
                               b_shift_limit=10 / 255,
                               p=0.3),
                ],
                p=0.2
            ),
            A.OneOf(
                [
                    # ガンマ変換
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    # ぼかし
                    A.Blur(p=0.6),
                    # シャープネス
                    A.IAASharpen(p=0.6),
                    # ガウスノイズ
                    A.GaussNoise(var_limit=(0.01, 0.05), mean=0, p=0.05),
                    # グレースケール
                    A.ToGray(p=0.05)
                ],
                p=0.2
            ),
            A.OneOf(
                [
                    # 水平に反転
                    A.HorizontalFlip(p=1),
                    # 垂直に反転
                    A.VerticalFlip(p=1),
                    # 転置
                    A.Transpose(p=1),
                    # 90°単位で回転
                    A.RandomRotate90(p=1)
                ],
                p=1
            ),
            # 霧をシミュレート
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, p=0.05),
            # リサイズ
            A.Resize(height=512, width=512, p=1),
            # 矩形領域の粗いDropout
            A.Cutout(num_holes=random.randint(1, 6),
                     max_h_size=64,
                     max_w_size=64,
                     fill_value=0,
                     p=0.15),
            # torch tensor
            ToTensorV2(p=5.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc',
                                 min_area=0,
                                 min_visibility=0,
                                 label_fields=['labels'])
    )


def get_train_transforms_detr():
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    return A.Compose(
        [
            # 水平に反転
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                MT.CustomRandomResize(scales, max_size=1333, p=0.5),
                A.Compose(
                    [
                        MT.CustomRandomResize([400, 500, 600]),
                        MT.CustomRandomSizedCrop(384, 600),
                        MT.CustomRandomResize(scales, max_size=1333)
                    ],
                    p=0.5
                )
            ]),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                        max_pixel_value=1),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc',
                                 min_area=0,
                                 min_visibility=0,
                                 label_fields=['labels'])
    )


def get_valid_transforms():
    return A.Compose([A.Resize(height=512,
                               width=512,
                               p=1.0),
                      ToTensorV2(p=1.0)],
                     p=1.0,
                     bbox_params=A.BboxParams(format='pascal_voc',
                                              min_area=0,
                                              min_visibility=0,
                                              label_fields=['labels']))


def get_valid_transforms_detr():
    return A.Compose(
        [
            MT.CustomRandomResize([800], max_size=1333),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                        max_pixel_value=1),
            ToTensorV2(),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format='pascal_voc',
                                 min_area=0,
                                 min_visibility=0,
                                 label_fields=['labels'])
    )


def get_test_transform():
    return A.Compose([A.Resize(height=512,
                               width=512,
                               p=1.0),
                      ToTensorV2(p=1.0)],
                     p=1.0
                     )
