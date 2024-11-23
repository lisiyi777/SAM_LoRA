import cv2
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from dataset import *
from model import *
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class InferSAM:
    def __init__(self, orig_path="sam_vit_b_01ec64.pth", tuned_path=None):
        self.device = "cuda"

        kwargs = {"rank": 4, "scale": 1}
        self.orig_sam = MyFastSAM(orig_path, **kwargs).to(self.device)
        if tuned_path is not None:
            self.tuned_sam = MyFastSAM.load_from_checkpoint(tuned_path, **kwargs).to(self.device)

        self.orig_predictor = SamPredictor(self.orig_sam.lora_sam)
        self.tuned_predictor = SamPredictor(self.tuned_sam.lora_sam)

    def inference_one(self, img, target, coords=None, labels=None, bboxes=None, tuned=True):
        # # preprocess
        # img = img.to(self.device)
        # if coords is not None:
        #     coords = coords.to(self.device)
        #     labels = labels.to(self.device).long()
        # if bboxes is not None:
        #     bboxes = bboxes.to(self.device)

        # # construct batched input
        # batches, target = self.orig_sam.construct_batched_input(img.unsqueeze(0), target.unsqueeze(0), prompt='box', max_boxes=20)

        # # forward
        # results = self.tuned_sam.lora_sam(batches, multimask_output=False)
        # pred_masks = results[0]["masks"].squeeze(1)
        # target = target[0]
        if tuned:
            predictor = self.tuned_predictor
        else:
            predictor = self.orig_predictor
        reverse = transforms.ToPILImage()
        img = np.array(reverse(img))
        predictor.set_image(img)

        pred_masks, pred_ious, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=None if bboxes is None else bboxes[None, :],
            multimask_output=True,
        )

        for i, mask in enumerate(pred_masks):
            true_count = mask.sum().item()
            if true_count == 0:
                print(f"Mask {i} is empty (all False).")
            else:
                print(f"Mask {i} contains {true_count} True pixels.")


        # sample masks
        if bboxes is not None:
            tgt_masks = self.tuned_sam.box_sample(target, bboxes)
            # pred_masks = self.tuned_sam.box_sample(pred_masks, bboxes)
        else:
            tgt_masks = self.tuned_sam.point_sample(target, coords, labels)
            # pred_masks = self.tuned_sam.point_sample(pred_masks, coords, labels)

        pred_masks = torch.from_numpy(pred_masks)        

        # calculate iou
        ious = self.calc_IoU(pred_masks, tgt_masks)

        # plot
        self.__plot(img, ious, pred_masks, tgt_masks, coords, labels, bboxes)


    def __plot(self, img, ious, masks, gt_masks, coords=None, labels=None, bboxes=None):
        plt.figure(figsize=(12, 8))
        # img = img.cpu().permute(1, 2, 0)
        masks = masks.cpu()
        gt_masks = gt_masks.cpu()
        # if labels is not None:
        #     labels = labels.cpu()
        #     coords = coords.cpu()
        # if bboxes is not None:
        #     bboxes = bboxes.cpu()
        

        for i, (score, mask, gt_mask) in enumerate(zip(ious, masks, gt_masks)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.imshow(mask, alpha=0.5)
            plt.title("Score {:.4f}".format(score))
            
            if bboxes is not None:
                x0, y0 = bboxes[0], bboxes[1]
                w, h = bboxes[2] - bboxes[0], bboxes[3] - bboxes[1]
                ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

            if labels is not None:
                marker_size = 120
                pos_points = coords[labels==1]
                neg_points = coords[labels==0]
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

            for i, gt_mask in enumerate(gt_masks):
                plt.subplot(3, 3, i+4)
                plt.imshow(img)
                plt.imshow(gt_mask, alpha=0.5)
                plt.title("Ground Truth")  

        plt.tight_layout()
        plt.show()

    def calc_IoU(self, pred, target):
        # pred = torch.from_numpy(pred).to(target.device).bool()
        pred = pred.to(target.device).bool()
        target = target.bool()

        intersections = torch.sum(pred & target, dim=(1, 2))
        unions = torch.sum(pred | target, dim=(1, 2))        
        epsilon = 1e-7
        ious = intersections.float() / (unions.float() + epsilon)        
        return ious
        
    def inference_all(self, image):
        image_bgr = image[[2, 1, 0], :, :]

        image_bgr_np = image_bgr.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
        image_bgr_np = (image_bgr_np * 255).astype(np.uint8)


        annotated_images = []
        mask_generator = SamAutomaticMaskGenerator(self.orig_sam.lora_sam)
        sam_result = mask_generator.generate(image_bgr_np)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=image_bgr_np.copy(), detections=detections)
        annotated_images.append(annotated_image)


        annotated_images2 = []
        mask_generator2 = SamAutomaticMaskGenerator(self.tuned_sam.lora_sam)
        sam_result2 = mask_generator2.generate(image_bgr_np)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections2 = sv.Detections.from_sam(sam_result=sam_result2)
        annotated_image2 = mask_annotator.annotate(scene=image_bgr_np.copy(), detections=detections2)
        annotated_images2.append(annotated_image2)

        images_to_plot = [image_bgr_np] + [img for img in annotated_images] + [img for img in annotated_images2]
        titles = ['Original'] + ['SAM'] + ['LoRA-SAM']

        sv.plot_images_grid(
            images=images_to_plot,
            grid_size=(1, len(images_to_plot)),
            titles=titles
        )

def GetData(high_res):
    if high_res:
        input_transform = transforms.Compose([
        transforms.Resize((800, 1280), antialias=True),
        transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 1280), antialias=True),
        ])
    else:
        input_transform = transforms.Compose([
        transforms.Resize((160, 256), antialias=True),
        transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 256), antialias=True),
        ])

    dataset = SA1B_Dataset("./data", transform=input_transform, target_transform=target_transform)
    return dataset

def inference_with_points(infer:InferSAM, high_res=False, tuned=True):
    # TODO: high res
    # input_point = np.array([[500,200],[300,200]])
    # input_label = np.array([0,1])

    dataset = GetData(high_res)

    image, target = dataset.__getitem__(55)
    print("Expect 160,256: ", image.shape)
    print("Expect 160,256: ", target.shape)

    kwargs = {}
    # input_point = torch.tensor([[100,75],[150,75]])
    # input_label = torch.tensor([0,1])
    # input_box = torch.tensor([40, 50, 130, 110])
    input_point = np.array([[100,75],[150,75]])
    input_label = np.array([0,1])
    input_box = np.array([40, 50, 130, 110])
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label
    # kwargs["bboxes"] = input_box
    kwargs["tuned"] = tuned
    
    infer.inference_one(image, target, **kwargs)

def inference_all(infer:InferSAM, high_res=False):
    if high_res:
        input_transform = transforms.Compose([
        transforms.Resize((800, 1280), antialias=True),
        transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 1280), antialias=True),
        ])
    else:
        input_transform = transforms.Compose([
        transforms.Resize((160, 256), antialias=True),
        transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 256), antialias=True),
        ])
    
    dataset = SA1B_Dataset("./data", transform=input_transform, target_transform=target_transform)
    image, target = dataset.__getitem__(122)
    image = TF.resize(image, size=[160, 256])
    target = TF.resize(target, size=[160, 256], interpolation=TF.InterpolationMode.NEAREST)
    if high_res:
        image = TF.resize(image, size=[800, 1280])
        target = TF.resize(target, size=[800, 1280], interpolation=TF.InterpolationMode.NEAREST)
    
    infer.inference_all(image)


if __name__ == "__main__":
    orig_path = ".\checkpoints\sam_vit_b_01ec64.pth"
    tuned_path = ".\logs\checkpoints\MyFastSAM-epoch=08-val_loss=1.1954.ckpt"
    infer = InferSAM(orig_path, tuned_path)

    # inference_with_points(infer, high_res=False, tuned=True)
    inference_all(infer)

    # # inference the original sam for the entire image
    # inference_all(checkpoint=orig_path, origin=True)
    # # inference the lora sam for the entire image
    # inference_all(checkpoint=tuned_path)
