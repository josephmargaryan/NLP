import os
import json
import torch
from PIL import Image
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_folder, annotations_folder, transforms=None):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.transforms = transforms
        self.image_paths = [
            f
            for f in os.listdir(images_folder)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.images_folder, img_name)
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        annotation_path = os.path.join(
            self.annotations_folder, f"{os.path.splitext(img_name)[0]}.json"
        )
        with open(annotation_path) as f:
            annotation = json.load(f)

        boxes = torch.tensor(annotation["boxes"], dtype=torch.float32)
        labels = torch.tensor(annotation["labels"], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_paths)


def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


images_folder = "path/to/dataset/images"
annotations_folder = "path/to/dataset/bounding_boxes"

dataset = CustomDataset(images_folder, annotations_folder, transforms=get_transform())
data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda x: tuple(zip(*x)),
)
