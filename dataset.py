import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from pycocotools import mask as mask_utils

import json
import numpy as np
import os
import glob
from tqdm import tqdm


class SA1B_Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, folder, preprocess_idx=None, **kwargs):
        img_folder = os.path.join(folder, "sa1b")
        super().__init__(img_folder, **kwargs)

        self.ws = os.path.join(folder, "preprocess")
        if preprocess_idx is not None:
            self.preprocess(start_index=preprocess_idx)

        self.tgt_files = glob.glob(os.path.join(self.ws, '*.pt'))
        self.tgt_files.sort()

        img_folder = os.path.join(img_folder, "noclass")
        img_files = [os.path.basename(file) for file in self.tgt_files]
        img_files = [file.replace(".pt", ".jpg") for file in img_files]
        self.img_files = [os.path.join(img_folder, file) for file in img_files]


    def preprocess(self, start_index=0):
        for file, _ in tqdm(self.imgs[start_index:]):
            masks = json.load(open(f'{file[:-3]}json'))['annotations']

            target = []
            for m in masks:
                # decode masks from COCO RLE format
                target.append(mask_utils.decode(m['segmentation']))

            target = np.stack(target, axis=-1)
            target[target > 0] = 1 # convert to binary masks

            tgt_file = os.path.basename(file).replace(".jpg", ".pt")
            tgt_file = os.path.join(self.ws, tgt_file)
            torch.save(target, tgt_file)
        
    def __getitem__(self, index):
        path = self.img_files[index]
        image = self.loader(path)
        masks = torch.load(self.tgt_files[index])
        masks[masks > 0] = 1 # convert to binary masks

        if self.transform is not None:
            image = self.transform(image)

        return image, masks
    def __len__(self):
        return len(self.tgt_files)
    

def collate_fn(batch):
    images, target = zip(*batch)
    return torch.stack(images, dim=0), target


def get_loaders(data_dir="..\..\SAM_LoRA\data", batch_size=8, preprocess_idx=None, use_small_subset=None, split_ratio=0.8):
    input_transform = transforms.Compose([
        transforms.Resize((160, 256), antialias=True),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 256), antialias=True),
    ])

    dataset = SA1B_Dataset(data_dir, preprocess_idx=preprocess_idx, transform=input_transform, target_transform=target_transform)
    
    if use_small_subset is not None:
        indices = torch.randperm(len(dataset))[:use_small_subset]
        dataset = torch.utils.data.Subset(dataset, indices)
    print("dataset size:\t", len(dataset))

    train_size = int(len(dataset) * split_ratio)
    # train_size=1
    test_size = len(dataset) - train_size

    torch.random.manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    num_workers = min(16, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, test_loader

