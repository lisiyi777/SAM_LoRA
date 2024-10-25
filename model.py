from segment_anything import sam_model_registry
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRASAM(nn.Module):
    def __init__(self, lora_rank: int, lora_scale: float, checkpoint="sam_vit_b_01ec64.pth", high_res = False):
        super().__init__()

        self.device = "cuda"
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        sam = sam.to(self.device)

        if not high_res:        
            sam.image_encoder.img_size = 256
            avg_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
            downsampled_tensor = avg_pooling(sam.image_encoder.pos_embed.permute(0,3,1,2)).permute(0,2,3,1)
            print("before downsampled_tensor",sam.image_encoder.pos_embed.data.shape)
            sam.image_encoder.pos_embed.data = downsampled_tensor
            sam.prompt_encoder.input_image_size = [256, 256]
            sam.prompt_encoder.image_embedding_size = [16, 16]
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
        
        self.sam = sam

    def point_sample(self, points_coords, points_label, all_masks):
        # all_masks: [N, H, W], one image, N masks
        # points_coords: (N, 2)
        # points_label: (N, 1), 1 for foreground, 0 for background
        # return: sampled_masks: [3, H, W], masks order from big to small
        # you can modify the signature of this function
        
        # find the target mask that contains the point
        mask_ids = []
        for i, mask in enumerate(all_masks):
            is_valid = True
            for is_fore, (x, y) in zip(points_label, points_coords):
                on_mask = mask[y][x]
                is_valid = (on_mask and is_fore) or (not on_mask and not is_fore)
                if not is_valid:
                    break

            if is_valid:
                mask_ids.append(i)

        # assign according to the size of the mask and leave one or two of the three empty.
        sampled_masks = torch.zeros((3, all_masks.shape[1], all_masks.shape[2]))
        mask_ids.sort(key=lambda i: all_masks[i].sum(), reverse=True)
        
        for i, idx in enumerate(mask_ids):
            sampled_masks[i] = all_masks[idx]
        
        return sampled_masks
