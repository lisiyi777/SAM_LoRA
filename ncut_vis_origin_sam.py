
from einops import rearrange
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
import numpy as np
from celeb_dataset import CelebAMask_Dataset
from dataset import SA1B_Dataset
import tqdm
from ncut_pytorch import NCUT, rgb_from_tsne_3d

DIM = 256
class SAM(torch.nn.Module):
    def __init__(self, checkpoint="checkpoints/sam_vit_b_01ec64.pth", **kwargs):
        super().__init__(**kwargs)
        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything.modeling.sam import Sam

        sam: Sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        from segment_anything.modeling.image_encoder import (
            window_partition,
            window_unpartition,
        )

        def new_block_forward(self, x: torch.Tensor) -> torch.Tensor:
            print(f"Input shape at block start: {x.shape}")
            shortcut = x
            x = self.norm1(x)
            print(f"After norm1 shape: {x.shape}")
            # Window partition
            if self.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, self.window_size)
                print(f"After window partition shape: {x.shape}, pad: {pad_hw}")

            x = self.attn(x)
            print(f"After attention shape: {x.shape}")

            # Reverse window partition
            if self.window_size > 0:
                x = window_unpartition(x, self.window_size, pad_hw, (H, W))
                print(f"After window unpartition shape: {x.shape}")

            self.attn_output = x.clone()
            print(f"Attention output shape: {self.attn_output.shape}")

            x = shortcut + x
            mlp_output = self.mlp(self.norm2(x))
            print(f"MLP output shape: {mlp_output.shape}")

            self.mlp_output = mlp_output.clone()
            x = x + mlp_output
            self.block_output = x.clone()
            print(f"Block output shape: {self.block_output.shape}")

            return x

        setattr(sam.image_encoder.blocks[0].__class__, "forward", new_block_forward)

        self.image_encoder = sam.image_encoder
        self.image_encoder.img_size = DIM
        patch_size =16
        embed_dim = 768
        self.image_encoder.pos_embed = nn.Parameter(
            torch.zeros(1, self.image_encoder.img_size // patch_size, self.image_encoder.img_size // patch_size, embed_dim)
        )
        self.image_encoder.eval()
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.image_encoder(x)

        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i, blk in enumerate(self.image_encoder.blocks):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)
            print(f"block {i} block_output shape: {blk.block_output.shape}")
        attn_outputs = torch.stack(attn_outputs)
        mlp_outputs = torch.stack(mlp_outputs)
        block_outputs = torch.stack(block_outputs)
        return attn_outputs, mlp_outputs, block_outputs


def image_sam_feature(
    images, resolution=(DIM, DIM), checkpoint="checkpoints/sam_vit_b_01ec64.pth"
):
    transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Resize(resolution,antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    feat_extractor = SAM(checkpoint=checkpoint)
    print("SAM model loaded.")
    attn_outputs, mlp_outputs, block_outputs = [], [], []
    for i, image in tqdm.tqdm(enumerate(images)):
        torch_image = transform(image)
        print(f"torch_image shape: {torch_image.shape}")
        attn_output, mlp_output, block_output = feat_extractor(
            torch_image.unsqueeze(0)
        )
        attn_outputs.append(attn_output.cpu())
        mlp_outputs.append(mlp_output.cpu())
        block_outputs.append(block_output.cpu())
    attn_outputs = torch.cat(attn_outputs, dim=1)
    mlp_outputs = torch.cat(mlp_outputs, dim=1)
    block_outputs = torch.cat(block_outputs, dim=1)

    return attn_outputs, mlp_outputs, block_outputs



input_transform = transforms.Compose([
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = CelebAMask_Dataset("./data",
                        transform=input_transform,
                        target_transform=target_transform)

# dataset = SA1B_Dataset("./data",
#                         transform=input_transform,
#                         target_transform=target_transform)
print("number of images in the dataset:", len(dataset))

images = [dataset[i][0] for i in range(10)]

attn_outputs, mlp_outputs, block_outputs = image_sam_feature(images)
print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
num_nodes = np.prod(attn_outputs.shape[1:4])

for i_layer in tqdm.tqdm(range(12)):

    attn_eig, _ = NCUT(num_eig=100, device="cpu").fit_transform(
        attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
    )
    _, attn_rgb = rgb_from_tsne_3d(attn_eig, device="cpu")
    attn_rgb = attn_rgb.reshape(attn_outputs[i_layer].shape[:3] + (3,))
    mlp_eig, _ = NCUT(num_eig=100, device="cpu").fit_transform(
        mlp_outputs[i_layer].reshape(-1, mlp_outputs[i_layer].shape[-1])
    )
    _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, device="cpu")
    mlp_rgb = mlp_rgb.reshape(mlp_outputs[i_layer].shape[:3] + (3,))
    block_eig, _ = NCUT(num_eig=100, device="cpu").fit_transform(
        block_outputs[i_layer].reshape(-1, block_outputs[i_layer].shape[-1])
    )
    _, block_rgb = rgb_from_tsne_3d(block_eig, device="cpu")
    block_rgb = block_rgb.reshape(block_outputs[i_layer].shape[:3] + (3,))

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(4, 10, figsize=(10, 5))
    for ax in axs.flatten():
        ax.axis("off")
    for i_col in range(10):
        img = images[i_col]
        img = transforms.Resize((DIM,DIM),antialias=True)(img)
        # print(f"img shape: {img.shape}")
        axs[0, i_col].imshow(img.permute(1, 2, 0))
        axs[1, i_col].imshow(attn_rgb[i_col])
        axs[2, i_col].imshow(mlp_rgb[i_col])
        axs[3, i_col].imshow(block_rgb[i_col])

    axs[1, 0].set_title("attention layer output", ha="left")
    axs[2, 0].set_title("MLP layer output", ha="left")
    axs[3, 0].set_title("sum of residual stream", ha="left")

    plt.suptitle(f"SAM layer {i_layer} NCUT spectral-tSNE", fontsize=16)
    # plt.show()
    save_dir = "./ncut_result_origin_256"
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/sam_layer_{i_layer}.jpg", bbox_inches="tight")
    plt.close()