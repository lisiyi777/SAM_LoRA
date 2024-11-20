import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from .lora import *
from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
from segment_anything import sam_model_registry
from copy import deepcopy
from typing import Any, Dict, List, Tuple


def point_sample(all_masks, points_coords, points_label):
    # all_masks: [N, H, W], one image, N masks
    # points_coords: (N, 2)
    # points_label: (N, 1), 1 for foreground, 0 for background
    # return: sampled_masks: [3, H, W], masks order from big to small
    # you can modify the signature of this function

    # TODO: what does points_label do???
    valid_masks = []
    for mask in all_masks:
        x = points_coords[:,0]
        y = points_coords[:,1]
        valid_points = mask[y][x]
        if torch.all(valid_points):
            valid_masks.append(mask)

    # sorting the masks based on the total number of non-zero pixels
    valid_masks.sort(key=lambda m: m.sum(), reverse=True)
    valid_masks = torch.stack(valid_masks)

    sampled_masks = torch.zeros((3, all_masks.shape[1], all_masks.shape[2]))
    if len(valid_masks) >= 3:
        sampled_masks = valid_masks[:3]
    else:
        sampled_masks[:len(valid_masks)] = valid_masks
    return sampled_masks

def box_sample(all_masks, bbox):
    # all_masks: [N, H, W], one image, N masks
    # bbox: (xxyy)
    # return: sampled_masks: [3, H, W], masks order from big to small
    # you can modify the signature of this function

    # Calculate IoUs
    bbox_mask = torch.zeros_like(all_masks, dtype=int)
    bbox_mask[:,bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    intersections = torch.logical_and(all_masks, bbox_mask).sum(dim=(1, 2))
    unions = torch.logical_or(all_masks, bbox_mask).sum(dim=(1, 2))
    ious = intersections.float() / unions.float()   # (N,)
    
    # TODO: Find the mask indices with the highest IoU and smaller size than bbox?
    sorted_mask_ids = torch.argsort(ious, descending=True)
    selected_masks = []
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    for mask_id in sorted_mask_ids:
        mask = all_masks[mask_id]
        if mask.sum().item() <= bbox_area:
            selected_masks.append(mask)
        if len(selected_masks) == 3:
            break

    while len(selected_masks) < 3:
        selected_masks.append(torch.zeros_like(all_masks[0]))

    # Stack and return the selected masks
    sampled_masks = torch.stack(selected_masks)
    return sampled_masks


class MyFastSAM(pl.LightningModule):
    def __init__(self, checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth", **kwargs):
        super().__init__()
        self.save_hyperparameters()  # Automatically saves all __init__ args, including kwargs

        self.orig_sam = self.__orig_sam(checkpoint)  # Original SAM model
        self.lora_sam = self.__lora_sam(**kwargs)  # LoRA-enhanced SAM model

        # Configurable hyperparameters from kwargs
        self.lora_rank = kwargs.get("lora_rank", 4)
        self.lora_scale = kwargs.get("lora_rank", 1)
        self.lr = kwargs.get("lr", 1e-4)
        self.linear = kwargs.get("linear", True)
        self.conv2d = kwargs.get("conv2d", False)
        
        
    def forward(self, batched_input, multimask_output=False):
        """
        Forward pass for LoRA-SAM.
        
        Args:
            batched_input (list[dict]): A list of input dictionaries, each containing:
                - 'image': torch.Tensor of shape [3, H, W], transformed for the model.
                - 'original_size': tuple (H, W), original image size before transformation.
                - 'point_coords': torch.Tensor of shape [B, N, 2], point coordinates.
                - 'point_labels': torch.Tensor of shape [B, N], labels for the points.
                - 'boxes': torch.Tensor of shape [B, 4], bounding box coordinates (optional).
            multimask_output (bool): Whether to output multiple disambiguating masks.

        Returns:
            list[dict]: A list of dictionaries, each containing:
                - 'masks': torch.Tensor of shape [B, C, H, W], binary mask predictions.
                - 'iou_predictions': torch.Tensor of shape [B, C], predicted mask IoUs.
                - 'low_res_logits': torch.Tensor of shape [B, C, H, W], low-res logits.
        """
        # Extract image features using LoRA-enhanced image encoder
        images = torch.stack([self.lora_sam.preprocess(x["image"]) for x in batched_input])  # [B, 3, H, W]
        image_features = self.lora_sam.image_encoder(images)
        # print('batched_input:' , batched_input[0])
        # Prepare prompts for the prompt encoder
        prompts = {
            "point_coords": [x.get("point_coords") for x in batched_input],
            "point_labels": [x.get("point_labels") for x in batched_input],
            "boxes": [x.get("boxes") for x in batched_input],
            "mask_inputs": [x.get("mask_inputs") for x in batched_input],
        }

        # Encode the prompts
        prompt_features = self.lora_sam.prompt_encoder(
            points=prompts["point_coords"],
            boxes=prompts["boxes"],
            masks=prompts["mask_inputs"],
        )

        # Decode the masks using the mask decoder
        masks, iou_predictions, low_res_logits = self.lora_sam.mask_decoder(
            image_features=image_features,
            image_embeddings=prompt_features,
            multimask_output=multimask_output,
        )

        # Prepare the results in the format expected by SAM
        results = []
        for i, input_dict in enumerate(batched_input):
            results.append({
                "masks": masks[i],  # Binary mask predictions
                "iou_predictions": iou_predictions[i],  # IoU predictions
                "low_res_logits": low_res_logits[i],  # Low-resolution logits
            })

        return results

    def __orig_sam(self, checkpoint, high_res=False):
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(self.device)

        # TODO: hack original sam to take low res images
        if not high_res:        
            sam.image_encoder.img_size = 256
            avg_pooling = nn.AvgPool2d(kernel_size=4, stride=4)
            downsampled_tensor = avg_pooling(sam.image_encoder.pos_embed.permute(0,3,1,2)).permute(0,2,3,1)
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
        return sam
        

    def __lora_sam(self, **kwargs):
        lora_sam = deepcopy(self.orig_sam)
        
        # Freeze original sam
        BaseFinetuning.freeze(lora_sam, train_bn=True)
        # for param in lora_sam.parameters():
        #     param.requires_grad_(False)
        
        # Inject LoRA        
        lora_sam = self.inject_lora(lora_sam, **kwargs)
        lora_sam = lora_sam.to(self.device)

        # Verify
        # self.check_lora_sam(lora_sam)

        return lora_sam

    def check_lora_sam(self, model):
        print("lora sam structure: \n", model)

        for name, param in model.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")

        model_parameters = filter(lambda p: True, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("total params: ", params)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("training params: ", params)

    def inject_lora(self, model, **kwargs):
        rank = kwargs.get("rank", 4)
        scale = kwargs.get("scale", 1)
        for name, block in model.named_children():  
            if isinstance(block, nn.Linear):
            # patch every nn.Linear in the model
               if kwargs.get("linear"):
                    block = MonkeyPatchLoRALinear(block, rank, scale)
                    setattr(model, name, block)
            # patch every nn.Conv2d in the model
            elif isinstance(block, nn.Conv2d):
                if kwargs.get("conv2d"):
                    block = MonkeyPatchLoRAConv2D(block, rank, scale)
                    setattr(model, name, block)
            # patch every nn.ConvTranspose2d in the model
            elif isinstance(block, nn.ConvTranspose2d):
                if kwargs.get("convtrans2d"):
                    block = MonkeyPatchLoRAConvTranspose2D(block, rank, scale)
                    setattr(model, name, block)
            #iterates over the immediate children of the model (not recursively)
            elif isinstance(block, nn.Module):
                self.inject_lora(block, **kwargs)
        return model
            

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=self.lr)
        return optimizer

    def calc_loss(self, prediction, targets):
        dice_loss = self.mask_dice_loss(prediction, targets)
        focal_loss = self.mask_focal_loss(prediction, targets)
        iou_loss = self.iou_token_loss(prediction, targets)
        return dice_loss + focal_loss + iou_loss

    @staticmethod
    def mask_dice_loss(prediction, targets, epsilon=1):
        prediction = torch.sigmoid(prediction)
        intersection = (prediction * targets).sum()
        cardinality = prediction.sum() + targets.sum()
        dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)
        dice_loss = 1 - dice_score
        return dice_loss

    @staticmethod
    def mask_focal_loss(prediction, targets):
        alpha=.25
        gamma=1.8
        alpha = torch.tensor([alpha, 1-alpha])
        BCE_loss = F.binary_cross_entropy_with_logits(prediction, targets, reduction='none')
        at = alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        focal_loss = at*(1-pt)**gamma * BCE_loss
        return focal_loss
    
    @staticmethod
    def iou_token_loss(iou_prediction, prediction, targets):
        mask_pred = (prediction >= 0.5).float()
        intersection = torch.sum(torch.mul(mask_pred, targets), dim=(-2, -1))
        union = torch.sum(mask_pred, dim=(-2, -1)) + torch.sum(targets, dim=(-2, -1)) - intersection
        epsilon = 1e-7
        batch_iou = intersection / (union + epsilon)
        batch_iou = batch_iou.unsqueeze(1)
        iou_loss = F.mse_loss(iou_prediction, batch_iou, reduction='mean')
        return iou_loss


    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        target = [mask.to(self.device) for mask in target]
        batched_input = self.construct_batched_input(images, targets)
        predictions = self.forward(batched_input)
        loss = self.calc_loss(predictions, targets)
        self.log('train_loss', loss, prog_bar=True)
        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images.to(self.device)
        targets = [mask.to(self.device) for mask in targets]
        batched_input = self.construct_batched_input(images, targets)
        predictions = self.forward(batched_input)
        loss = self.calc_loss(predictions, targets)
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def construct_batched_input(self, images, targets):
        # 1a. single point prompt training
        # 1b. iterative point prompt training up to 3 iteration
        # 2. box prompt training, only 1 iteration
        # TODO: 2 different ways to construct inputs
        """
        Constructs the batched input for the SAM model from images and target masks.

        Args:
            images (torch.Tensor): A tensor of shape [B, 3, H, W], where B is the batch size.
            targets (list[torch.Tensor]): A list of length B, where each element is a tensor
                                        of shape [N, H, W] (N masks per image).

        Returns:
            list[dict]: A batched input list compatible with SAM's forward pass.
        """
        batched_input = []
        # (Point Prompt Training)
        for img, target in zip(images, targets):
            # Randomly sample one point per mask
            N, H, W = target.shape
            mask_idxs = torch.arange(N)
            point_coords = []
            point_labels = []

            for idx in mask_idxs:
                mask = target[idx]
                # Sample a single foreground point
                fg_points = torch.nonzero(mask)  # Coordinates of foreground pixels
                if len(fg_points) > 0:
                    point_coords.append(fg_points[torch.randint(len(fg_points), (1,))].squeeze(0))
                    point_labels.append(1)  # Foreground label

                # Sample a single background point
                bg_points = torch.nonzero(mask == 0)  # Coordinates of background pixels
                if len(bg_points) > 0:
                    point_coords.append(bg_points[torch.randint(len(bg_points), (1,))].squeeze(0))
                    point_labels.append(0)  # Background label

            # Convert to tensors and normalize point coordinates to match the input size
            point_coords = torch.stack(point_coords, dim=0).float() if point_coords else torch.empty(0, 2)
            point_labels = torch.tensor(point_labels).float() if point_labels else torch.empty(0)

            # Construct the input dictionary
            input_dict = {
                "image": img,
                "original_size": (H, W),
                "point_coords": point_coords.unsqueeze(0),  # [1, N, 2]
                "point_labels": point_labels.unsqueeze(0),  # [1, N]
            }

            # Box Prompt Training (Optional)
            if self.linear:
                # Use bounding boxes derived from target masks
                x, y = torch.where(target.any(0))  # Get bounding box for the mask
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                input_dict["boxes"] = torch.tensor([[x_min, y_min, x_max, y_max]]).float()

            batched_input.append(input_dict)

        return batched_input


