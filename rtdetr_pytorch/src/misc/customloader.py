from ultralytics import YOLO
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
import os
from PIL import Image

def PotholeDataset(VisionDataset):
    def __init__(self, root, dataset_type, transform=None, target_transform=None):
        super(PotholeDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type
        self.data = []
        self.targets = []
        self.classes = ['pothole']
        self.load_data()
    
    def load_data(self):
        if self.dataset_type == 'train':
            self.image_path = os.path.join(self.root, 'images', 'train')
            self.label_path = os.path.join(self.root, 'labels', 'train')
        elif self.dataset_type == 'val':
            self.image_path = os.path.join(self.root, 'images', 'val')
            self.label_path = os.path.join(self.root, 'labels', 'val')
        else:
            raise ValueError('Invalid dataset type')
        self.labels = os.listdir(self.label_path)

        for labels in self.labels:
            with open(os.path.join(self.label_path, labels), 'r') as f:
                for line in f:
                    class_id, x, y, w, h = line.strip().split()
                    self.data.append(os.path.join(self.image_path, labels.replace('.txt', '.jpg')))
                    self.targets.append([int(class_id), float(x), float(y), float(w), float(h)])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target