# from lora_sam import *
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import SA1B_Dataset
from celeb_dataset import CelebAMask_Dataset
from ncut_pytorch import NCUT, rgb_from_tsne_3d
from model_ncut import MyFastSAM
import os
import torch
import tqdm

device = 'cpu'

def ncut_visualize(images, checkpoint):
    kwargs = {"lora_rank": 4, "lora_scale": 1}
    if checkpoint is not None:
            feat_extractor = MyFastSAM.load_from_checkpoint(checkpoint, **kwargs).to(device)

    attn_outputs, mlp_outputs, block_outputs = [], [], []
    for i, image in enumerate(images):
        attn_output, mlp_output, block_output = feat_extractor(
            image.unsqueeze(0).to(device),
        )
        attn_outputs.append(attn_output.cpu())
        mlp_outputs.append(mlp_output.cpu())
        block_outputs.append(block_output.cpu())
    attn_outputs = torch.cat(attn_outputs, dim=1)
    mlp_outputs = torch.cat(mlp_outputs, dim=1)
    block_outputs = torch.cat(block_outputs, dim=1)
    return attn_outputs, mlp_outputs, block_outputs
    
def visualize_feature_maps(feature_maps, title="Feature Map"):
    num_maps = feature_maps.shape[1]
    fig, axes = plt.subplots(1, num_maps, figsize=(num_maps * 2.5, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(feature_maps[0, i].detach().numpy(), cmap='viridis')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()



if __name__ == "__main__":
    W = 256
    H = 256
    checkpoint= "./checkpoints/MyFastSAM-epoch=06-val_loss=30.8608.ckpt"
    high_res = False


    input_transform = transforms.Compose([
        transforms.Resize((H, W), antialias=True),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((H, W), antialias=True),
    ])
    print('dataset init')
    dataset = CelebAMask_Dataset("./data",
                           transform=input_transform,
                           target_transform=target_transform)
    # dataset = SA1B_Dataset("./data",
    #                         transform=input_transform,
    #                         target_transform=target_transform)
    # generate a batch of images
    images = []
    for i in range(10):
        image, target = dataset.__getitem__(i)
        images.append(image)
    images = torch.stack(images, dim=0)

    # get the features
    attn_outputs, mlp_outputs, block_outputs=ncut_visualize(images, checkpoint=checkpoint)
    print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)

    # visualize_feature_maps(block_outputs[-1], "Attention Maps")
    num_eig = 100
    for i_layer in tqdm.tqdm(range(12)):
        attn_eig, _ = NCUT(num_eig=num_eig, device=device).fit_transform(
            attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
        )
        _, attn_rgb = rgb_from_tsne_3d(attn_eig, device=device)
        attn_rgb = attn_rgb.reshape(attn_outputs[i_layer].shape[:3] + (3,))

        mlp_eig, _ = NCUT(num_eig=num_eig, device=device).fit_transform(
            mlp_outputs[i_layer].reshape(-1, mlp_outputs[i_layer].shape[-1])
        )
        _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, device=device)
        mlp_rgb = mlp_rgb.reshape(mlp_outputs[i_layer].shape[:3] + (3,))
        
        block_eig, _ = NCUT(num_eig=num_eig, device=device).fit_transform(
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
        save_dir = "./ncut_result_tuned"
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/sam_layer_{i_layer}.jpg", bbox_inches="tight")
        plt.close()


