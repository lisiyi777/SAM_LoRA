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

        kwargs = {"linear": True, "conv2d": False}
        self.orig_sam = MyFastSAM(orig_path, **kwargs).to(self.device)
        if tuned_path is not None:
            self.tuned_sam = MyFastSAM.load_from_checkpoint(tuned_path, **kwargs).to(self.device)

        self.orig_predictor = SamPredictor(self.orig_sam.lora_sam)
        self.tuned_predictor = SamPredictor(self.tuned_sam.lora_sam)

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

            for i, gt_mask in enumerate(gt_masks):
                plt.subplot(3, 3, i+7)
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
    input_label = np.array([1,1])
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

def save_fig(fig, save_dir, batch_idx):
    """Save the figure with a unique name based on the batch index."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"batch_{batch_idx}.png")
    fig.savefig(save_path)
    print(f"Figure saved at {save_path}")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=350):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[0, 0], pos_points[0, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def test(infer: InferSAM, val_loader):
    model = infer.tuned_sam
    device = infer.device
    save_dir = "./output_plots"
    model.eval()
    with torch.no_grad():
        ious = torch.tensor([], device=device)
        for batch_idx, (image, target) in tqdm(enumerate(val_loader)):
            image = image.to(device)
            target = [mask.to(device) for mask in target]
            batched_input, target = model.construct_batched_input(image, target)
            target = torch.stack(target, dim=0)
            h, w = target.shape[-2:]
            predictions = model.forward(batched_input, multimask_output=False)

            pred_mask = [p["masks"].squeeze(1) for p in predictions]         
            for p, t in zip(pred_mask, target):
                iou = infer.calc_IoU(p, t)
            
                ious = torch.cat([ious, ious])
        
            masks = [pred_mask[0].cpu().numpy(), target[0].cpu().numpy()]
            image_cpu = batched_input[0]["image"].cpu().permute(1, 2, 0)
            box_cpu = batched_input[0]["boxes"].cpu()
            # print(ious.shape)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            titles = ["Predicted Masks", "Ground Truth Masks"]

            for axis, mask in zip(axes, masks):
                axis.imshow(image_cpu)
                for m in mask:
                    show_mask(m, axis, random_color=True)
                for b in box_cpu:
                    show_box(b, axis)
                axis.set_title(f"mean IoU: {ious.mean().item():.4f}")

            # Set IoU in the figure title
            fig.suptitle(f"Batch {batch_idx} - IoU: {total_ious.mean().item():.4f}")
            
            save_fig(fig, save_dir, batch_idx)

        mean_ious = total_ious.mean()
        print("TEST total masks {} mIoU {}".format(len(total_ious), mean_ious.item()))

if __name__ == "__main__":
    orig_path = ".\checkpoints\sam_vit_b_01ec64.pth"
    tuned_path = ".\checkpoints\MyFastSAM-epoch=05.ckpt"
    infer = InferSAM(orig_path, tuned_path)
    train_loader, val_loader = get_loaders(
        data_dir="./data",
        batch_size=1,
        use_small_subset=30
    )

    # inference_with_points(infer, high_res=False, tuned=True)
    # inference_all(infer)
    test(infer, val_loader)

