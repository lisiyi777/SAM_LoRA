# from lora_sam import *
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from lora_sam.plotter import Plotter
from lora_sam.dataset import SA1B_Dataset

from ncut_pytorch import NCUT, rgb_from_tsne_3d
from ncut_pytorch.backbone import load_model, extract_features
from lora_sam.model import MyFastSAM

import os
from torchvision.datasets import ImageFolder
import torch

device = 'cpu'

def ncut_visualize(images):
    # model = load_model(model_name="SAM(sam_vit_b)")
    kwargs = {"lora_rank": 4, "lora_scale": 1, "high_res":high_res}
    feat_extractor = MyFastSAM(**kwargs).to(device)
    attn_outputs, mlp_outputs, block_outputs = [], [], []
    for i, image in enumerate(images):
        # feat = feat_extractor(torch_image.unsqueeze(0).cuda()).cpu()
        attn_output, mlp_output, block_output = feat_extractor(
            image.unsqueeze(0).to(device),
        )
        # feats.append(feat)
        attn_outputs.append(attn_output.cpu())
        mlp_outputs.append(mlp_output.cpu())
        block_outputs.append(block_output.cpu())
    attn_outputs = torch.cat(attn_outputs, dim=1)
    mlp_outputs = torch.cat(mlp_outputs, dim=1)
    block_outputs = torch.cat(block_outputs, dim=1)

    # feats = torch.cat(feats, dim=1)
    # feats = rearrange(feats, "l b c h w -> l b h w c")
    return attn_outputs, mlp_outputs, block_outputs

def inference(checkpoint="sam_vit_b_01ec64.pth"):
    high_res = True
    if high_res:
        input_transform = transforms.Compose([
            transforms.Resize((800, 1280), antialias=True),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((800, 1280), antialias=True),
        ])
        input_box = np.array([40, 50, 130, 110])
        input_point = np.array([[25, 25], [25, 25]])
        input_label = np.array([0, 1])
    else:
        input_transform = transforms.Compose([
            transforms.Resize((160, 256), antialias=True),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 256), antialias=True),
        ])
        input_box = np.array([40, 50, 130, 110])
        input_point = np.array([[25, 25], [25, 25]])
        input_label = np.array([0, 1])

    print('plotter init')
    plotter = Plotter(checkpoint, high_res=high_res)
    print('dataset init')
    dataset = SA1B_Dataset("./data",
                           transform=input_transform,
                           target_transform=target_transform)
    #train_loader, test_loader = get_loaders(batch_size=8)

    image, target = dataset.__getitem__(1)  #[3, 800, 1280])
    image = TF.resize(image, size=[160, 256])
    target = TF.resize(target,
                       size=[160, 256],
                       interpolation=TF.InterpolationMode.NEAREST)

    kwargs = {}
    kwargs["bboxes"] = input_box
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label

    # images, target = next(iter(test_loader))
    print('inference_plot starts')
    plotter.inference_plot(image, target, **kwargs)

if __name__ == "__main__":
    # inference()
    checkpoint="./logs_points/checkpoints/MyFastSAM-epoch=25-val_loss=1.1973.ckpt"
    high_res = False
    if high_res:
        input_transform = transforms.Compose([
            transforms.Resize((800, 1280), antialias=True),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((800, 1280), antialias=True),
        ])
        input_box = np.array([40, 50, 130, 110])
        input_box = None
        input_point = np.array([[25, 25], [25, 25]])
        input_label = np.array([0, 1])
    else:
        input_transform = transforms.Compose([
            transforms.Resize((160, 256), antialias=True),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 256), antialias=True),
        ])
        input_box = np.array([40, 50, 130, 110])
        input_box = None
        input_point = np.array([[25, 25], [25, 25]])
        input_label = np.array([0, 1])

    kwargs = {}
    kwargs["bboxes"] = input_box
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label
    print('dataset init')
    dataset = SA1B_Dataset("./data",
                           transform=input_transform,
                           target_transform=target_transform)

    # generate a batch of images
    images = []
    for i in range(10):
        image, target = dataset.__getitem__(i)  #[3, 800, 1280])
        print(image.shape)
        images.append(image)
    images = torch.stack(images, dim=0)

    # get the features
    attn_outputs, mlp_outputs, block_outputs=ncut_visualize(images)
    # ncut_visualize(images,**kwargs)

    print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
    # num_nodes = np.prod(attn_outputs.shape[1:4])


    i_layer = 11

    for i_layer in range(12):

        attn_eig, _ = NCUT(num_eig=100, device=device).fit_transform(
            attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
        )
        _, attn_rgb = rgb_from_tsne_3d(attn_eig, device=device)
        attn_rgb = attn_rgb.reshape(attn_outputs[i_layer].shape[:3] + (3,))

        mlp_eig, _ = NCUT(num_eig=100, device=device).fit_transform(
            mlp_outputs[i_layer].reshape(-1, mlp_outputs[i_layer].shape[-1])
        )
        _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, device=device)
        mlp_rgb = mlp_rgb.reshape(mlp_outputs[i_layer].shape[:3] + (3,))
        
        block_eig, _ = NCUT(num_eig=100, device=device).fit_transform(
            block_outputs[i_layer].reshape(-1, block_outputs[i_layer].shape[-1])
        )
        _, block_rgb = rgb_from_tsne_3d(block_eig, device=device)
        block_rgb = block_rgb.reshape(block_outputs[i_layer].shape[:3] + (3,))

        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(4, 10, figsize=(10, 5))
        for ax in axs.flatten():
            ax.axis("off")
        for i_col in range(10):
            axs[0, i_col].imshow(images[i_col].permute(1, 2, 0))
            axs[1, i_col].imshow(attn_rgb[i_col])
            axs[2, i_col].imshow(mlp_rgb[i_col])
            axs[3, i_col].imshow(block_rgb[i_col])

        axs[1, 0].set_title("attention layer output", ha="left")
        axs[2, 0].set_title("MLP layer output", ha="left")
        axs[3, 0].set_title("sum of residual stream", ha="left")

        plt.suptitle(f"SAM layer {i_layer} NCUT spectral-tSNE", fontsize=16)
        # plt.show()
        save_dir = "./ncut_result"
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/sam_layer_{i_layer}.jpg", bbox_inches="tight")
        plt.close()
