import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from lora import *
import pytorch_lightning as pl
from segment_anything import sam_model_registry
from copy import deepcopy
import random
from utils import *
import matplotlib.pyplot as plt

class MyFastSAM(pl.LightningModule):
    def __init__(self, checkpoint: str = "../sam_vit_b_01ec64.pth", **kwargs):
        super().__init__()
        orig_sam = self.__orig_sam(checkpoint)
        self.lora_sam = self.__lora_sam(orig_sam, **kwargs)

        self.lora_rank = kwargs.get("lora_rank", 4)
        self.lora_scale = kwargs.get("lora_rank", 1)
        self.lr = kwargs.get("lr", 1e-4)
        self.linear = kwargs.get("linear", True)
        self.conv2d = kwargs.get("conv2d", False)
        
    def __orig_sam(self, checkpoint, high_res=False):
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

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
        BaseFinetuning.freeze(lora_sam, train_bn=True)

        # Inject LoRA
        # lora_sam = self.inject_lora(lora_sam, **kwargs)

        replace_LoRA(lora_sam.mask_decoder, MonkeyPatchLoRALinear)
        # if self.linear:
        #     replace_LoRA(lora_sam.mask_decoder, MonkeyPatchLoRALinear)
        # if self.conv2d:
        #     replace_LoRA(lora_sam, MonkeyPatchLoRAConv2D)

        self.check_lora_sam(lora_sam)

        return lora_sam

    def check_lora_sam(self, model, print_all=False):
        if print_all:
            print("lora sam structure: \n", model)
            for name, param in model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

        model_parameters = filter(lambda p: True, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("total params: ", params)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("training params: ", params)

    def forward(self, batched_input, multimask_output=False):
        # Extract image features using LoRA-enhanced image encoder
        # device = next(self.parameters()).device
        images = torch.stack([self.lora_sam.preprocess(x["image"]) for x in batched_input])  # [B, 3, H, W]

        image_features = self.lora_sam.image_encoder(images)
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
                masks=None,
            )

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
            if not self.training:
                masks = masks > 0.0

            # Prepare the results in the format expected by SAM
            results.append({
                "masks": masks,  # Binary mask predictions
                "iou_predictions": iou_predictions,  # IoU predictions
                "low_res_logits": low_res_masks,  # Low-resolution logits
            })

        return results
    
    def training_step(self, batch, batch_idx):
        image, target = batch
        batched_input, target = self.construct_batched_input(image, target)
        # batched_input: a list (len=batch_size) of dics: 'img':3,160,256, 'boxes':16,4
        # batched_targets: a list (len=batch_size) of (16,160,256)
        target = torch.stack(target, dim=0)
        # target:8, 16, 160, 256
        # boxes:8, 16, 4
        # images:8, 3, 160, 256
        h, w = target.shape[-2:]
        pred = self.forward(batched_input, multimask_output=False)
        pred = torch.stack([i["masks"] for i in pred], dim=0).cuda()    #8,16,1,160,256
        pred = pred.reshape(-1, 1, h, w)

        focal_loss, dice_loss = self.calc_loss(pred, target)
        loss = focal_loss + 0.01 * dice_loss
        self.log('train_dice_loss', dice_loss, prog_bar=True)
        self.log('train_focal_loss', focal_loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    # TODO:
        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3

    def validation_step(self, batch, batch_idx):
        image, target = batch
        batched_input, target = self.construct_batched_input(image, target)
        target = torch.stack(target, dim=0)
        h, w = target.shape[-2:]
        predictions = self.forward(batched_input, multimask_output=False)
        pred = torch.stack([i["masks"] for i in predictions], dim=0)   #8,16,1,160,256
        pred = pred.reshape(-1, 1, h, w)

        focal_loss, dice_loss = self.calc_loss(pred, target)
        loss = focal_loss + 0.01 * dice_loss
        self.log('val_dice_loss', dice_loss, prog_bar=True)
        self.log('val_focal_loss', focal_loss, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        pred_mask = [p["masks"].squeeze(1) for p in predictions]         
        for p, t in zip(pred_mask, target):
            ious = calc_IoU(p, t)

        self.log('val_iou', ious.mean(), prog_bar=True)
        print(f"Valildation mean IoU: {ious.mean():.4f}")
        return loss

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(lora_parameters, lr=self.lr)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def construct_batched_input(self, images, targets, prompt='box', max_boxes=16):
        # 1a. single point prompt training
        # 1b. iterative point prompt training up to 3 iteration
        # 2. box prompt training, only 1 iteration
        batched_input = []
        updated_targets = []
        device = images[0].device

        for img, target in zip(images, targets):
            # Randomly sample one point per mask
            N, H, W = target.shape
            mask_idxs = torch.arange(N, device=device)

            if prompt == 'point':
                point_coords = []
                point_labels = []
                for idx in mask_idxs:
                    mask = target[idx]
                    # Sample a single foreground point
                    fg_points = torch.nonzero(mask, as_tuple=False).to(device)
                    if len(fg_points) > 0:
                        point_coords.append(fg_points[torch.randint(len(fg_points), (1,), device=device)].squeeze(0))
                        point_labels.append(1)  # Foreground label

                    # Sample a single background point
                    bg_points = torch.nonzero(mask == 0, as_tuple=False).to(device)  # Ensure device consistency
                    if len(bg_points) > 0:
                        point_coords.append(bg_points[torch.randint(len(bg_points), (1,), device=device)].squeeze(0))
                        point_labels.append(0)  # Background label

                # Convert to tensors and normalize point coordinates to match the input size
                point_coords = torch.stack(point_coords, dim=0).float().to(device) if point_coords else torch.empty(0, 2, device=device)
                point_labels = torch.tensor(point_labels, device=device).float() if point_labels else torch.empty(0, device=device)

                # Construct the input dictionary
                input_dict = {
                    "image": img.cuda(),
                    "original_size": (H, W),
                    "point_coords": point_coords.unsqueeze(0),  # [1, N, 2]
                    "point_labels": point_labels.unsqueeze(0),  # [1, N]
                }

            # Box Prompt Training
            if prompt == 'box':
                boxes, updated_target = self.generate_box_prompts(target, max_boxes=max_boxes, device=device)
                input_dict = {
                    "image": img.cuda(),
                    "original_size": (updated_target.shape[1], updated_target.shape[2]),
                    "boxes": boxes.cuda(),
                }

                updated_targets.append(updated_target)

            batched_input.append(input_dict)

        return batched_input, updated_targets

    def generate_box_prompts(self, target, max_boxes, device):
        non_empty_masks = [mask for mask in target if mask.sum() > 0]

        num_samples = min(len(non_empty_masks), max_boxes)
        selected_masks = random.sample(non_empty_masks, num_samples)

        boxes = []
        updated_targets = []
        # for mask in selected_masks:
        #     y, x = torch.where(mask > 0)
        #     x_min, x_max = x.min().item(), x.max().item()
        #     y_min, y_max = y.min().item(), y.max().item()
        #     boxes.append([x_min, y_min, x_max, y_max])
        #     updated_targets.append(mask)
        for mask in selected_masks:
            mask_y, mask_x = torch.where(mask > 0)
            x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)
            delta_w = min(random.random() * 0.2 * w, 20)
            delta_h = min(random.random() * 0.2 * h, 20)

            x1, y1, x2, y2  = center_x - (w + delta_w) / 2, center_y - (h + delta_h) / 2, \
                                center_x + (w + delta_w) / 2, center_y + (h + delta_h) / 2
            boxes.append([x1, y1, x2, y2])
            updated_targets.append(mask)

        # Pad results if there are fewer than max_boxes masks
        while len(boxes) < max_boxes:
            # Add an empty box and an empty mask
            boxes.append([0, 0, 0, 0])
            updated_targets.append(torch.zeros_like(target[0]))

        boxes = torch.tensor(boxes, dtype=torch.float, device=device)
        updated_targets = torch.stack(updated_targets, dim=0).to(device)
        return boxes, updated_targets

    def calc_loss(self, prediction, targets):
        device = 'cuda'

        h, w = targets.shape[-2:]
        targets = targets.reshape(-1, h, w)
        prediction = prediction.squeeze(1)

        focal_loss = torch.tensor(0., device=device)
        dice_loss = torch.tensor(0., device=device)
        for pred, target in zip(prediction, targets):
            dice_loss += mask_dice_loss(pred, target)
            focal_loss += mask_focal_loss(pred, target)
        return dice_loss, focal_loss

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

    def point_sample(self,all_masks, points_coords, points_label):
        # all_masks: [N, H, W], one image, N masks
        # points_coords: (N, 2)
        # points_label: (N,), 1 for foreground, 0 for background
        # return: sampled_masks: [3, H, W], masks order from big to small
        # you can modify the signature of this function

        # valid_masks = all_masks
        valid_masks = []
        for mask in all_masks:
            x = points_coords[:,0]
            y = points_coords[:,1]
            on_mask = mask[y, x].bool()  # Check if points are on the mask

            # Validate points using the corrected logic
            valid_points = (on_mask & points_label==1) | (~on_mask & points_label==0)
            if torch.all(valid_points):
                valid_masks.append(mask)

        # sorting the masks based on the total number of non-zero pixels
        if len(valid_masks) > 0:
            valid_masks.sort(key=lambda m: m.sum(), reverse=True)
            valid_masks = torch.stack(valid_masks)

        sampled_masks = torch.zeros((3, all_masks.shape[1], all_masks.shape[2]))
        if len(valid_masks) >= 3:
            sampled_masks = valid_masks[:3]
        elif len(valid_masks) > 0:
            sampled_masks[:len(valid_masks)] = valid_masks
        return sampled_masks
        
    def box_sample(self,all_masks, bbox):
        # all_masks: [N, H, W], one image, N masks
        # bbox: (xxyy)
        # return: sampled_masks: [3, H, W], masks order from big to small
        # you can modify the signature of this function

        print(all_masks.shape)

        # Calculate IoUs
        bbox_mask = torch.zeros_like(all_masks, dtype=int)
        bbox_mask[:,bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        intersections = torch.logical_and(all_masks, bbox_mask).sum(dim=(1, 2))
        unions = torch.logical_or(all_masks, bbox_mask).sum(dim=(1, 2))
        ious = intersections.float() / unions.float()   # (N,)
        
        # Find the mask indices with the highest IoU
        sorted_mask_ids = torch.argsort(ious, descending=True)
        selected_masks = []
        for mask_id in sorted_mask_ids:
            mask = all_masks[mask_id]
            selected_masks.append(mask)
            if mask.sum() == 0:
                print("@@@@@@@@@@@@@")
            else:
                print(mask.sum())
            if len(selected_masks) == 3:
                break

        while len(selected_masks) < 3:
            selected_masks.append(torch.zeros_like(all_masks[0]))

        # Stack and return the selected masks
        sampled_masks = torch.stack(selected_masks)
        return sampled_masks

    def construct_inference_input(self, images, targets, prompt='box'):
        batched_input = []
        device = images[0].device

        for img, target in zip(images, targets):
            # Randomly sample one point per mask
            N, H, W = target.shape
            mask_idxs = torch.arange(N, device=device)

            if prompt == 'point':
                point_coords = []
                point_labels = []
                for idx in mask_idxs:
                    mask = target[idx]
                    # Sample a single foreground point
                    fg_points = torch.nonzero(mask, as_tuple=False).to(device)
                    if len(fg_points) > 0:
                        point_coords.append(fg_points[torch.randint(len(fg_points), (1,), device=device)].squeeze(0))
                        point_labels.append(1)  # Foreground label

                    # Sample a single background point
                    bg_points = torch.nonzero(mask == 0, as_tuple=False).to(device)  # Ensure device consistency
                    if len(bg_points) > 0:
                        point_coords.append(bg_points[torch.randint(len(bg_points), (1,), device=device)].squeeze(0))
                        point_labels.append(0)  # Background label

                # Convert to tensors and normalize point coordinates to match the input size
                point_coords = torch.stack(point_coords, dim=0).float().to(device) if point_coords else torch.empty(0, 2, device=device)
                point_labels = torch.tensor(point_labels, device=device).float() if point_labels else torch.empty(0, device=device)

                # Construct the input dictionary
                input_dict = {
                    "image": img.cuda(),
                    "original_size": (H, W),
                    "point_coords": point_coords.unsqueeze(0),  # [1, N, 2]
                    "point_labels": point_labels.unsqueeze(0),  # [1, N]
                }

            # Box Prompt Training
            if prompt == 'box':
                boxes, updated_target = self.generate_inference_prompts(target, max_boxes=max_boxes, device=device)
                input_dict = {
                    "image": img.cuda(),
                    "original_size": (updated_target.shape[1], updated_target.shape[2]),
                    "boxes": boxes.cuda(),
                }

            batched_input.append(input_dict)

        return batched_input
    
    def generate_inference_prompts(self, target, device):
        non_empty_masks = [mask for mask in target if mask.sum() > 0]

        boxes = []
        for mask in non_empty_masks:
            mask_y, mask_x = torch.where(mask > 0)
            x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
            # center_x = (x1 + x2) / 2
            # center_y = (y1 + y2) / 2
            # w = (x2 - x1)
            # h = (y2 - y1)
            # delta_w = min(random.random() * 0.2 * w, 20)
            # delta_h = min(random.random() * 0.2 * h, 20)

            # x1, y1, x2, y2  = center_x - (w + delta_w) / 2, center_y - (h + delta_h) / 2, \
            #                     center_x + (w + delta_w) / 2, center_y + (h + delta_h) / 2
            boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(boxes, dtype=torch.float, device=device)
        return boxes

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
    alpha = torch.tensor([alpha, 1 - alpha], device=prediction.device)
    BCE_loss = F.binary_cross_entropy_with_logits(prediction.float(), targets.float(), reduction='none')

    at = alpha.gather(0, targets.view(-1).long()).view_as(targets)  # Match shape of targets
    pt = torch.exp(-BCE_loss)  # Probability of correct class
    focal_loss = at * (1 - pt) ** gamma * BCE_loss  # Focal loss formula

    return focal_loss.mean()

@staticmethod
def iou_token_loss(iou_prediction, prediction, targets):
    mask_pred = (prediction >= 0.).float()
    intersection = torch.sum(torch.mul(mask_pred, targets), dim=(-2, -1))
    union = torch.sum(mask_pred, dim=(-2, -1)) + torch.sum(targets, dim=(-2, -1)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)
    batch_iou = batch_iou.unsqueeze(1)
    iou_loss = F.mse_loss(iou_prediction, batch_iou, reduction='mean')
    return iou_loss
