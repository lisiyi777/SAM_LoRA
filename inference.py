import cv2
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from dataset import *
from model import *
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from collections import defaultdict
from celeb_dataset import get_celeb_loaders
import time
from utils import *

class InferSAM:
    def __init__(self, orig_path="sam_vit_b_01ec64.pth", tuned_path=None):
        self.device = "cuda"

        kwargs = {"linear": True, "conv2d": False}
        self.orig_sam = MyFastSAM(orig_path, **kwargs).to(self.device)
        if tuned_path is not None:
            self.tuned_sam = MyFastSAM.load_from_checkpoint(tuned_path, **kwargs).to(self.device)
            self.tuned_predictor = SamPredictor(self.tuned_sam.lora_sam)

        self.orig_predictor = SamPredictor(self.orig_sam.lora_sam)

    def inference_one(self, img, target, coords=None, labels=None, bboxes=None, tuned=True):
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
        else:
            tgt_masks = self.tuned_sam.point_sample(target, coords, labels)

        pred_masks = torch.from_numpy(pred_masks)        

        ious = self.calc_IoU(pred_masks, tgt_masks)

        self.__plot(img, ious, pred_masks, tgt_masks, coords, labels, bboxes)


    def __plot(self, img, ious, masks, gt_masks, coords=None, labels=None, bboxes=None):
        plt.figure(figsize=(12, 8))
        # img = img.cpu().permute(1, 2, 0)
        masks = masks.cpu()
        gt_masks = gt_masks.cpu()

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

            for i, gt_mask in enumerate(gt_masks):
                plt.subplot(3, 3, i+7)
                plt.imshow(img)
                plt.imshow(gt_mask, alpha=0.5)
                plt.title("Ground Truth")  

        plt.tight_layout()
        plt.show()

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

def GetData(high_res=False):
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
    # input_point = np.array([[500,200],[300,200]])
    # input_label = np.array([0,1])

    dataset = GetData(high_res)

    image, target = dataset.__getitem__(65)
    print("Expect 160,256: ", image.shape)
    print("Expect 160,256: ", target.shape)

    kwargs = {}
    # input_point = torch.tensor([[100,75],[150,75]])
    # input_label = torch.tensor([0,1])
    # input_box = torch.tensor([40, 50, 130, 110])
    input_point = np.array([[50,30],[150,75]])
    input_label = np.array([1,1])
    input_box = np.array([40, 50, 130, 110])
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label
    # kwargs["bboxes"] = input_box
    kwargs["tuned"] = tuned
    
    infer.inference_one(image, target, **kwargs)

def sam_inference_all(infer:InferSAM, high_res=False):
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
    image, target = dataset.__getitem__(88)
    image = TF.resize(image, size=[160, 256])
    target = TF.resize(target, size=[160, 256], interpolation=TF.InterpolationMode.NEAREST)
    if high_res:
        image = TF.resize(image, size=[800, 1280])
        target = TF.resize(target, size=[800, 1280], interpolation=TF.InterpolationMode.NEAREST)
    
    infer.inference_all(image)

def save_fig(fig, save_dir, batch_idx):
    """Save the figure with a unique name based on the batch index."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"batch_{batch_idx}.png")
    fig.savefig(save_path)
    print(f"Figure saved at {save_path}")


def inference_all(infer: InferSAM, val_loader):
    # model = infer.orig_sam
    model = infer.tuned_sam
    device = infer.device
    save_dir = "./output_plots"
    model.eval()
    total_time = 0.0
    total_masks = 0
    
    with torch.no_grad():
        ious = torch.tensor([], device=device)
        for batch_idx, (image, target) in tqdm(enumerate(val_loader)):
            image = image.to(device)*255.
            target = [mask.to(device) for mask in target]
            batched_input = model.construct_inference_input_all(image, target)
            target = torch.stack(target, dim=0)
            h, w = target.shape[-2:]

            start_time = time.time()
            
            predictions = model.forward(batched_input, multimask_output=False)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            num_masks = sum([len(p["masks"]) for p in predictions])
            total_masks += num_masks

            pred_mask = [p["masks"].squeeze(1) for p in predictions]         
            for p, t in zip(pred_mask, target):
                iou = infer.calc_IoU(p, t)
            
                ious = torch.cat([ious, iou])
        
            masks = [pred_mask[0].cpu().numpy(), target[0].cpu().numpy()]
            image_cpu = batched_input[0]["image"].cpu().permute(1, 2, 0)
            # print(ious.shape)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            titles = ["Predicted Masks", "Ground Truth Masks"]

            for axis, mask in zip(axes, masks):
                axis.imshow(image_cpu/255.)
                for m in mask:
                    show_mask(m, axis, random_color=True)
                axis.set_title(f"mean IoU: {ious.mean().item():.4f}")
            
            save_fig(fig, save_dir, batch_idx)

        avg_time_per_mask = total_time / total_masks
        print(f"Average Annotation Time per Mask: {avg_time_per_mask:.4f}s")
        mean_ious = ious.mean()
        print(f"TEST mIoU {mean_ious.item()}")

def split_masks_by_size(infer: InferSAM, val_loader):
    iou_results = defaultdict(list)
    model = infer.tuned_sam
    device = infer.device
    save_dir = "./output_plots"
    model.eval()
    with torch.no_grad():
        for batch_idx, (image, target) in tqdm(enumerate(val_loader)):
            image = image.to(device)*255.
            target = [mask.to(device) for mask in target]
            
            large_inputs, medium_inputs, small_inputs, target_large, target_medium, target_small = model.construct_inference_input(image, target)
            
            if len(target_large) != 0:
                large_preds = model.forward(large_inputs, multimask_output=False)
            if len(target_medium) != 0:
                medium_preds = model.forward(medium_inputs, multimask_output=False)
            if len(target_small) != 0:
                small_preds = model.forward(small_inputs, multimask_output=False)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            titles = ["Large Predictions", "Medium Predictions", "Small Predictions"]
            
            large_preds = torch.stack([i["masks"] for i in large_preds], dim=0)    #bz,num_mask,1,160,256
            medium_preds = torch.stack([i["masks"] for i in medium_preds], dim=0)    #bz,num_mask,1,160,256
            small_preds = torch.stack([i["masks"] for i in small_preds], dim=0)    #bz,num_mask,1,160,256
            batch_size, _, _, h, w = large_preds.shape
            large_preds = large_preds.reshape(batch_size, -1, h, w)
            medium_preds = medium_preds.reshape(batch_size, -1, h, w)
            small_preds = small_preds.reshape(batch_size, -1, h, w)

            for p, t in zip(small_preds, target_small):
                iou = infer.calc_IoU(p, t)
                iou_results["small"].extend(iou.cpu().tolist())
            for p, t in zip(medium_preds, target_medium):
                iou = infer.calc_IoU(p, t)
                iou_results["medium"].extend(iou.cpu().tolist())
            for p, t in zip(large_preds, target_large):
                iou = infer.calc_IoU(p, t)
                iou_results["large"].extend(iou.cpu().tolist())

            for idx, (pred, target, title) in enumerate(zip(
                [large_preds, medium_preds, small_preds],
                [target_large, target_medium, target_small],
                titles,
            )):
                axes[0, idx].imshow((image[0]/255.).cpu().permute(1, 2, 0))
                for m in pred[0]:
                    show_mask(m.cpu().numpy(), axes[0, idx], random_color=True)
                axes[0, idx].set_title(title)

                axes[1, idx].imshow((image[0]/255.).cpu().permute(1, 2, 0))
                for m in target:
                    show_mask(m.cpu().numpy(), axes[1, idx], random_color=True)
                axes[1, idx].set_title(f"{title} Ground Truth")

            save_fig(fig, save_dir, f"batch_{batch_idx}.png")
            plt.close(fig)

    for size, iou_list in iou_results.items():
        mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        print(f"Mean IoU for {size}: {mean_iou:.4f}")
    
    # Overall mIoU
    all_ious = [iou for iou_list in iou_results.values() for iou in iou_list]
    overall_miou = sum(all_ious) / len(all_ious) if all_ious else 0
    print(f"Overall Mean IoU: {overall_miou:.4f}")

if __name__ == "__main__":
    orig_path = ".\checkpoints\sam_vit_b_01ec64.pth"
    # tuned_path = ".\checkpoints\Linear-epoch=00-val_iou=0.7465.ckpt"
    tuned_path = ".\checkpoints\MyFastSAM-epoch=00-val_loss=59.4682.ckpt"
    infer = InferSAM(orig_path, tuned_path)
    train_loader, val_loader = get_loaders(
        data_dir="./data",
        batch_size=1,
        use_small_subset=80
    )

    # inference_with_points(infer, high_res=False, tuned=True)
    # sam_inference_all(infer)
    inference_all(infer, val_loader)
    # split_masks_by_size(infer, val_loader)