import torch
import torchvision
from pycocotools import mask as mask_utils

import json
import numpy as np
import os
import glob
from tqdm import tqdm
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from utils import *
import os
import matplotlib.pyplot as plt        

class CelebAMask_Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, folder, preprocess_idx=None, **kwargs):
        img_folder = os.path.join(folder, "CelebAMask-HQ/CelebA-HQ-img")
        super().__init__(img_folder, **kwargs)

        self.ws = os.path.join(folder, "CelebAMask-HQ/CelebAMask-HQ-mask")
        if preprocess_idx is not None:
            self.__preprocess(start_index=preprocess_idx)

        # Use glob.glob to get a list of all files matching the pattern
        print("ws:\t", self.ws)
        self.tgt_files = glob.glob(os.path.join(self.ws, '*.pt'))
        self.tgt_files.sort()

        img_files = [os.path.basename(file) for file in self.tgt_files]
        img_files = [file.replace(".pt", ".jpg") for file in img_files]
        self.img_files = [os.path.join(img_folder,'noclass', file) for file in img_files]
    
    
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
        # print(type(masks))
        #11,512,512,3
        if masks.shape[-1] == 3:  # 检查最后一维是否为 3
            masks = masks[..., 0]  # 假设所有通道相同，取第一个通道即可 11,512,512
        masks_binary = (masks > 0).astype(int)
        # masks_binary = np.any(masks > 0, axis=1).astype(int)    #before: 3,11,512
        # print("masks_binary shape:\t", masks_binary.shape)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            masks = [self.target_transform(mask) for mask in masks_binary]
            # print("masks shape:\t", masks[0].shape)
            # print("mask len:\t", len(masks))    
            masks = torch.stack(masks, dim=0)

        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # ms = masks.cpu().numpy()
        # for m in ms:
        #     show_mask(m, axes[0], random_color=True)
        # save_fig(fig)

        # print("target shape:\t", masks.shape)
        masks = masks.reshape(-1,256,256)
        return image, masks
    

    def __len__(self):
        return len(self.tgt_files)
    
def save_fig(fig, save_dir = "./output_plots"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"aaaaaaa.png")
    fig.savefig(save_path)
    print(f"Figure saved at {save_path}")

def collate_fn(batch):
    images, target = zip(*batch)
    return torch.stack(images, dim=0), target

def get_celeb_loaders(data_dir="..\..\SAM_LoRA\data", batch_size=8, preprocess_idx=None, use_small_subset=None):
    input_transform = transforms.Compose([
        transforms.Resize((256, 256), antialias=True),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])

    dataset = CelebAMask_Dataset(data_dir, preprocess_idx=preprocess_idx, transform=input_transform, target_transform=target_transform)
    

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




# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="SA1B Dataset Loader with Preprocessing Index")
    # parser.add_argument("--folder", type=str, default="..\..\SAM_LoRA\data", help="Base folder for the dataset")
    # parser.add_argument("--preprocess_idx", type=int, help="Start index for preprocessing")
    # args = parser.parse_args()

    # # train_loader, test_loader = get_loaders(folder=args.folder, batch_size=2, preprocess_idx=args.preprocess_idx)
    # train_loader, test_loader = get_loaders(folder=args.folder, batch_size=2, preprocess_idx=6694)
    # images, target = next(iter(train_loader))
    # print("image shape:\t", images.shape)
    # print("len(target):\t", len(target))
    # print("target[0].shape:\t", target[0].shape)
