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

################## Model Utils ##################
def print_params(model):
  model_parameters = filter(lambda p: True, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print("total params: ", params)
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print("training params: ", params)

def __inject_lora(model, rank=4, scale=1, linear=True, conv2d=False):
    # TODO: inject according to the class
    if linear:
        for name, block in model.named_children():
            # patch every nn.Linear in the model
            if isinstance(block, nn.Linear):
                block = MonkeyPatchLoRALinear(block, rank, scale)
                setattr(model, name, block)

    if conv2d:
        for name, block in model.named_children():
            # patch every nn.Conv2d in the model
            if isinstance(block, nn.Conv2d):
                block = MonkeyPatchLoRAConv2D(block, rank, scale)
                setattr(model, name, block)

################## Model Utils ##################


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
    def __init__(self, lora_rank: int, lora_scale: float, checkpoint="sam_vit_b_01ec64.pth"):
        super().__init__()
        self.device = 'cuda'
        self.orig_sam = self.__orig_sam(checkpoint)
        self.lora_sam = self.__lora_sam()

        self.lr = 1e-5

    def forward(self, *args, **kwargs):
        """
        comments imported from original SAM code

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        return self.lora_sam(...)

    def __orig_sam(self, checkpoint, high_res=False):
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(self.device)
        sam.image_encoder.img_size = 256

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
        

    def __lora_sam(self):
        lora_sam = deepcopy(self.orig_sam)
        BaseFinetuning.freeze(lora_sam, train_bn=True)
        lora_sam = __inject_lora(lora_sam).to(self.device)
        return lora_sam

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        optimizer = torch.optim.AdamW(lora_parameters, lr=self.lr)
        return optimizer

    def calc_loss(self, prediction, targets):
        ...

    @staticmethod
    def mask_dice_loss(prediction, targets):
        ...

    @staticmethod
    def mask_focal_loss(prediction, targets):
        ...

    @staticmethod
    def iou_token_loss(iou_prediction, prediction, targets):
        ...

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
        target = [mask.to(self.device) for mask in target]
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
        target_masks = []
        target_boxes = []
        for i in range(len(images)):
            for _ in range(10):
                bbox = self.random_sample_bbox(images.shape)
                masks = self.box_sample(bbox, targets[i])
                if any((mask == 0).all() for mask in masks):
                    continue
                
            target_boxes.append(bbox)
            target_masks.append(masks.to(self.device))

        target_boxes = torch.Tensor(target_boxes).to(self.device)
        return images, target_boxes
