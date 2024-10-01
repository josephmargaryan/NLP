import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection

# Assuming you have your custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, annotations, transforms=None):
        # image_paths is a list of file paths to images
        # annotations is a list of dicts, where each dict contains boxes and labels for that image
        self.image_paths = image_paths
        self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        annotation = self.annotations[idx]
        boxes = torch.tensor(annotation["boxes"], dtype=torch.float32)
        labels = torch.tensor(annotation["labels"], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.image_paths)

# Example Transformations
import torchvision.transforms as T

def get_transform():
    return T.Compose([
        T.ToTensor(),  # Converts PIL image to a tensor
    ])

# Instantiate Dataset and DataLoader
dataset = CustomDataset(image_paths, annotations, transforms=get_transform())
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
