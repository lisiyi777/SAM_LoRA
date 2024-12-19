import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from lora import *
from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
from segment_anything import sam_model_registry
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import random
from inference import show_box, show_mask, save_fig
import matplotlib.pyplot as plt


class MyFastSAM(pl.LightningModule):
    def __init__(self, checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth", **kwargs):
        super().__init__()
        orig_sam = self.__orig_sam(checkpoint)
        self.lora_sam = self.__lora_sam(orig_sam, **kwargs) 

        # Configurable hyperparameters from kwargs
        self.lora_rank = kwargs.get("lora_rank", 4)
        self.lora_scale = kwargs.get("lora_rank", 1)
        self.lr = kwargs.get("lr", 1e-4)
        self.linear = kwargs.get("linear", True)
        self.conv2d = kwargs.get("conv2d", False)
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

        
        setattr(self.lora_sam.image_encoder.blocks[0].__class__, "forward", new_block_forward)
    
    def forward(self, batched_input):
        print("image shape: ", batched_input[0].shape)
        x = torch.stack([self.lora_sam.preprocess(x) for x in batched_input])
        self.lora_sam.image_encoder(x)
        attn_outputs, mlp_outputs, block_outputs = [], [], []
        # print("self.lora_sam.image_encoder.blocks length: ", len(self.lora_sam.image_encoder.blocks))
        for i, blk in enumerate(self.lora_sam.image_encoder.blocks):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)
            # print(f"block {i} attn_output shape: {blk.attn_output.shape}")
            # print(f"block {i} mlp_output shape: {blk.mlp_output.shape}")
            print(f"block {i} block_output shape: {blk.block_output.shape}")
        attn_outputs = torch.stack(attn_outputs)
        mlp_outputs = torch.stack(mlp_outputs)
        block_outputs = torch.stack(block_outputs)
        return attn_outputs, mlp_outputs, block_outputs

    def __orig_sam(self, checkpoint, high_res=False):
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        patch_size = 16
        embed_dim = 768
        # TODO: hack original sam to take low res images
        if not high_res: 
            img_size = 256
            img_embed_size = img_size // patch_size
            sam.image_encoder.img_size = img_size
            sam.image_encoder.pos_embed = nn.Parameter(
                torch.zeros(1, img_embed_size, img_embed_size, embed_dim)
            )
            sam.prompt_encoder.input_image_size = [img_size, img_size]
            sam.prompt_encoder.image_embedding_size = [img_embed_size, img_embed_size]
        else:
            sam.image_encoder.img_size = 1280
            target_embedding_size = (80,80)
            pos_embed = sam.image_encoder.pos_embed.data
            upscaled_pos_embed = F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=target_embedding_size,
                mode='bilinear',
                align_corners=False,
            ).permute(0, 2, 3, 1)
            sam.image_encoder.pos_embed = nn.Parameter(upscaled_pos_embed)
            sam.prompt_encoder.input_image_size = [1280, 1280]
            sam.prompt_encoder.image_embedding_size = [80, 80]
        return sam

    def __lora_sam(self, orig_sam, **kwargs):
        lora_sam = deepcopy(orig_sam)
        BaseFinetuning.freeze(lora_sam, train_bn=True)
        replace_LoRA(lora_sam.mask_decoder, MonkeyPatchLoRALinear)
        return lora_sam

