import numpy as np


# plt
from PIL import Image

# Pytorch
import torch
from torch.utils.data import Dataset
import datasets.transforms as T
import torchvision.transforms as V

# import custom_datasets.transforms_vision as TT


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

        image = Image.open(
            f'{self.image_dir}/{image_id}.jpg').convert('RGB')
        w, h = image.size
        # DETR takes in data in yolo format
        boxes = records[['x', 'y', 'w', 'h']].values

        # Area of bb
        area = boxes[:, 2] * boxes[:, 3]
        area = torch.as_tensor(area, dtype=torch.float32)

        # [x,y,w,h] -> [xmin,ymin,xmax,ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clip(min=0, max=w)
        boxes[:, 1::2].clip(min=0, max=h)

        # AS pointed out by PRVI It works better if the main class is labelled
        # as zero
        labels = np.zeros(len(boxes), dtype=np.int32)

        iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)

        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.long)
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

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

        image = Image.open(
            f'{self.image_dir}/{image_id}.jpg')  # .convert('RGB')
        im_w, im_h = image.size
        ratio = im_h / im_w
        image = image.resize((800, int(800 * ratio))).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_train_transforms():
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ])
        ),
        T.Compose([
            T.ToTensor(),
            # (*) convert [xmin,ymin,xmax,ymax] -> [cx,cy,width,height]
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ])


def get_valid_transforms():
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ])


def get_test_transform():
    return V.Compose([
        V.ToTensor(),
        V.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
