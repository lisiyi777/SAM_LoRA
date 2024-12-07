import torch
import torchvision
from pycocotools import mask as mask_utils

import json
import numpy as np
import os
import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class SA1B_Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, folder, preprocess_idx=None, **kwargs):
        img_folder = os.path.join(folder, "sa1b")
        super().__init__(img_folder, **kwargs)

        self.ws = os.path.join(folder, "preprocess")
        if preprocess_idx is not None:
            self.__preprocess(start_index=preprocess_idx)

        # Use glob.glob to get a list of all files matching the pattern
        self.tgt_files = glob.glob(os.path.join(self.ws, '*.pt'))
        self.tgt_files.sort()

        img_folder = os.path.join(img_folder, "noclass")
        img_files = [os.path.basename(file) for file in self.tgt_files]
        img_files = [file.replace(".pt", ".jpg") for file in img_files]
        self.img_files = [os.path.join(img_folder, file) for file in img_files]


    def __preprocess(self, start_index=0):
        # os.mkdir(self.ws)
        for file, _ in tqdm(self.imgs[start_index:]):
        # for file, _ in tqdm(self.imgs[:]):
            masks = json.load(open(f'{file[:-3]}json'))['annotations'] # load json masks

            target = []
            for m in masks:
                # decode masks from COCO RLE format
                target.append(mask_utils.decode(m['segmentation']))

            target = np.stack(target, axis=-1)
            target[target > 0] = 1 # convert to binary masks

            tgt_file = os.path.basename(file).replace(".jpg", ".pt")
            tgt_file = os.path.join(self.ws, tgt_file)

            if self.target_transform is not None:
                target = self.target_transform(target)

            torch.save(target, tgt_file)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
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


def get_loaders(data_dir="..\..\SAM_LoRA\data", batch_size=8, preprocess_idx=None, use_small_subset=None):
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
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size

    torch.random.manual_seed(1)
    num_workers = min(16, batch_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, test_loader

