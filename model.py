import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from lora_sam.lora import *
from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
from segment_anything import sam_model_registry
from copy import deepcopy
from typing import Any, Dict, List, Tuple
import random



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

        orig_sam = self.__orig_sam(checkpoint)  # Original SAM model
        self.lora_sam = self.__lora_sam(orig_sam, **kwargs)  # LoRA-enhanced SAM model

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
        # device = next(self.parameters()).device
        # print("batched_input in forward:", len(batched_input))
        images = torch.stack([self.lora_sam.preprocess(x["image"]) for x in batched_input])  # [B, 3, H, W]

        image_features = self.lora_sam.image_encoder(images)
        # print("image_features in forward:", image_features.shape)
        results = []
        for image_record, img_embedding in zip(batched_input, image_features):

            # Encode the prompts
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.lora_sam.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            # print("sparse_embeddings in forward:", sparse_embeddings.shape)
            # print("dense_embeddings in forward:", dense_embeddings.shape)
            # Decode the masks using the mask decoder
            low_res_masks, iou_predictions = self.lora_sam.mask_decoder(
                image_embeddings=img_embedding.unsqueeze(0),
                image_pe=self.lora_sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.lora_sam.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            # TODO: 0.0?
            # if not self.training:
            masks = masks > 0.0

            # Prepare the results in the format expected by SAM
            # print("masks in forward:", masks.squeeze(1).shape)
            results.append({
                "masks": masks.squeeze(1),  # Binary mask predictions
                "iou_predictions": iou_predictions,  # IoU predictions
                "low_res_logits": low_res_masks,  # Low-resolution logits
            })
        # print("results in forward:", len(results))
        return results

    def __orig_sam(self, checkpoint, high_res=False):
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

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
        

    def __lora_sam(self, orig_sam, **kwargs):
        lora_sam = deepcopy(orig_sam)
        
        # Freeze original sam
        BaseFinetuning.freeze(lora_sam, train_bn=True)
        # for param in lora_sam.parameters():
        #     param.requires_grad_(False)
        
        # Inject LoRA
        lora_sam = self.inject_lora(lora_sam, **kwargs)

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
        iou_preds = [x["iou_predictions"] for x in prediction]
        predictions = [x["masks"] for x in prediction]
        device = predictions[0].device  
        targets = [target.to(device) for target in targets]  

        focal_loss = torch.tensor(0., device=device)
        dice_loss = torch.tensor(0., device=device)
        iou_loss = torch.tensor(0., device=device)        
        for pred, target, iou_pred in zip(predictions, targets, iou_preds):
            dice_loss += 0.1 * self.mask_dice_loss(pred, target)
            focal_loss += self.mask_focal_loss(pred, target)
            iou_loss += self.iou_token_loss(iou_pred,pred, target)
        return dice_loss + focal_loss + iou_loss

    @staticmethod
    def mask_dice_loss(prediction, targets, epsilon=1):
        prediction = torch.sigmoid(prediction)
        intersection = (prediction * targets).sum()
        cardinality = prediction.sum() + targets.sum()
        dice_score = (2. * intersection + epsilon) / (cardinality + epsilon)
        dice_loss = 1 - dice_score
        return torch.mean(dice_loss)
    
    @staticmethod
    def mask_focal_loss(prediction, targets, alpha=0.25, gamma=1.8):
        prediction = prediction.squeeze(1)  # Remove the extra dimension if needed

        alpha = torch.tensor([alpha, 1 - alpha], device=prediction.device)
        BCE_loss = F.binary_cross_entropy_with_logits(prediction.float(), targets.float(), reduction='none')

        at = alpha.gather(0, targets.view(-1).long()).view_as(targets)  # Match shape of targets
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        focal_loss = at * (1 - pt) ** gamma * BCE_loss  # Focal loss formula

        return focal_loss.mean()
    
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
        batched_input, batched_targets = self.construct_batched_input(images, targets)
        predictions = self.forward(batched_input)
        loss = self.calc_loss(predictions, batched_targets)
        self.log('train_loss', loss, prog_bar=True)
        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = images
        targets = [mask for mask in targets]
        batched_input, batched_targets = self.construct_batched_input(images, targets)
        predictions = self.forward(batched_input)
        loss = self.calc_loss(predictions, batched_targets)
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def construct_batched_input(self, images, targets, prompt='point', max_masks=10):
        # 1a. single point prompt training
        # 1b. iterative point prompt training up to 3 iteration
        # 2. box prompt training, only 1 iteration
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
        updated_targets = []
        device = images.device
        for img, target in zip(images, targets):
            # Randomly sample one point per mask
            _, H, W = target.shape
            if prompt == 'point':
                point_coords, point_labels, updated_target = self.generate_point_prompts(target, max_masks=max_masks, device=device)
                # print("updated_target:", updated_target.shape) # [B, H, W]
                # print("point_coords:", point_coords.shape) # [B, N, 2]
                # print("point_labels:", point_labels.shape)   # [N, 2]
                input_dict = {
                    "image": img,
                    "original_size": (updated_target.shape[1], updated_target.shape[2]),
                    "point_coords": point_coords,
                    "point_labels": point_labels 
                }
            # Box Prompt Training
            if prompt == 'box':
                boxes, updated_target = self.generate_box_prompts(target, max_boxes=max_masks, device=device)
                boxes[:, 0::2] /= W
                boxes[:, 1::2] /= H
                # print("boxes:", boxes.shape)
                input_dict = {
                    "image": img,
                    "original_size": (updated_target.shape[1], updated_target.shape[2]),
                    "boxes": boxes,
                }
            # print("updated_target:",updated_target.shape)
            batched_input.append(input_dict)
            updated_targets.append(updated_target)
        return batched_input, updated_targets
    
    def point_sample(self, all_masks, points_coords, points_label):
        # all_masks: [N, H, W], one image, N masks
        # points_coords: (N, 2)
        # points_label: (N, 1), 1 for foreground, 0 for background
        # return: sampled_masks: [3, H, W], masks order from big to small
        # you can modify the signature of this function
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

    def generate_box_prompts(self, target, max_boxes, device):
        # Step 1: 过滤掉空白掩码
        non_empty_masks = [mask for mask in target if mask.sum() > 0]

        # Step 2: 随机选择部分掩码
        num_samples = min(len(non_empty_masks), max_boxes)
        selected_masks = random.sample(non_empty_masks, num_samples)

        # Step 3: 为每个掩码生成边框
        boxes = []
        updated_targets = []
        for mask in selected_masks:
            x, y = torch.where(mask > 0)
            x_min, x_max = x.min().item(), x.max().item()
            y_min, y_max = y.min().item(), y.max().item()
            boxes.append([x_min, y_min, x_max, y_max])
            updated_targets.append(mask)
        # Pad results if there are fewer than max_boxes masks
        while len(boxes) < max_boxes:
            # Add an empty box and an empty mask
            boxes.append([0, 0, 0, 0])
            updated_targets.append(torch.zeros_like(target[0]))

        boxes = torch.tensor(boxes, dtype=torch.float, device=device)
        updated_targets = torch.stack(updated_targets, dim=0).to(device)
        return boxes, updated_targets

    def generate_point_prompts(self, target, max_masks, device):
        non_empty_masks = [mask for mask in target if mask.sum() > 0]
        num_samples = min(len(non_empty_masks), max_masks)
        selected_masks = random.sample(non_empty_masks, num_samples)

        mask_point_coords = []
        mask_point_labels = []
        mask_updated_targets = []

        for mask in selected_masks:
            fg_points = torch.nonzero(mask, as_tuple=False)
            bg_points = torch.nonzero(mask == 0, as_tuple=False)
            points = []
            labels = []

            if len(fg_points) > 0:
                fg_index = torch.randint(len(fg_points), (1,), device=device)
                fg_point = fg_points[fg_index].squeeze(0)
                points.append(fg_point)
                labels.append(1)  # Foreground label
            else:
                # 为前景添加无效占位符
                points.append(torch.tensor([0, 0], device=device))
                labels.append(-1)

            if len(bg_points) > 0:
                bg_index = torch.randint(len(bg_points), (1,), device=device)
                bg_point = bg_points[bg_index].squeeze(0)
                points.append(bg_point)
                labels.append(0)  # Background label
            else:
                # 为背景添加无效占位符
                points.append(torch.tensor([0, 0], device=device))
                labels.append(-1)

            mask_point_coords.append(torch.stack(points))
            mask_point_labels.append(torch.tensor(labels, device=device))
            mask_updated_targets.append(mask)

        # 填充剩余的mask到max_masks
        while len(mask_point_coords) < max_masks:
            mask_point_coords.append(torch.tensor([[0, 0], [0, 0]], device=device))  # 添加无效坐标占位符
            mask_point_labels.append(torch.tensor([0, 0], device=device))
            zero_target = torch.zeros_like(target[0], device=device)
            mask_updated_targets.append(zero_target)

        point_coords = torch.stack(mask_point_coords)
        point_labels = torch.stack(mask_point_labels)
        updated_targets = torch.stack(mask_updated_targets)
        # print("updated_targets:", updated_targets.shape)
        # print("point_coords:", point_coords.shape)
        # print("point_labels:", point_labels.shape)
        return point_coords, point_labels, updated_targets
