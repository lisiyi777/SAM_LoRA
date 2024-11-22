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
    def __init__(self, checkpoint="sam_vit_b_01ec64.pth", origin=False):
        self.device = "cuda"

        kwargs = {"rank": 4, "scale": 1}
        if origin:
            self.sam = MyFastSAM(checkpoint, **kwargs).to(self.device)
        else:
            self.sam = MyFastSAM.load_from_checkpoint(checkpoint, **kwargs).to(self.device)
        self.predictor = SamPredictor(self.sam.lora_sam)

    def inference_one(self, img, target, coords=None, labels=None, bboxes=None):
        print("image shape ",img.shape)
        reverse = transforms.ToPILImage()
        img = np.array(reverse(img))
        print("image shape ",img.shape)
        self.predictor.set_image(img)

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=None if bboxes is None else bboxes[None, :],
            multimask_output=True,
        )

        if bboxes is not None:
            gt_masks = self.sam.box_sample(bboxes, target)
        else:
            gt_masks = self.sam.point_sample(target, coords, labels)

        self.__plot(img, scores, masks, gt_masks, coords, labels, bboxes)

    def __plot(self, img, scores, masks, gt_masks, coords=None, labels=None, bboxes=None):
        plt.figure(figsize=(12, 8))

        for i, (score, mask, gt_mask) in enumerate(zip(scores, masks, gt_masks)):
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

        plt.tight_layout()
        plt.show()

def inference(checkpoint = "sam_vit_b_01ec64.pth", origin=False):
    high_res = False
    if high_res:
        input_transform = transforms.Compose([
        transforms.Resize((800, 1280), antialias=True),
        transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 1280), antialias=True),
        ])
        input_point = np.array([[500,200],[300,200]])
        input_label = np.array([0,1])
    else:
        input_transform = transforms.Compose([
        transforms.Resize((160, 256), antialias=True),
        transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 256), antialias=True),
        ])
        input_point = np.array([[100,40],[150,40]])
        input_label = np.array([0,1])

    dataset = SA1B_Dataset("./data", transform=input_transform, target_transform=target_transform)
    image, target = dataset.__getitem__(55) #[3, 800, 1280])
    image = TF.resize(image, size=[160, 256])
    target = TF.resize(target, size=[160, 256], interpolation=TF.InterpolationMode.NEAREST)
    if high_res:
        image = TF.resize(image, size=[800, 1280])
        target = TF.resize(target, size=[800, 1280], interpolation=TF.InterpolationMode.NEAREST)

    kwargs = {}
    kwargs["coords"] = input_point
    kwargs["labels"] = input_label
    
    infer = InferSAM(checkpoint, origin)
    infer.inference_one(image, target, **kwargs)

def inference_all(checkpoint = "sam_vit_b_01ec64.pth", origin=False):
    high_res = True
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
    
    device = "cuda"

    kwargs = {"lora_rank": 4, "lora_scale": 1}
    if origin:
        sam = MyFastSAM(checkpoint, **kwargs).to(device)
    else:
        sam = MyFastSAM.load_from_checkpoint(checkpoint, **kwargs).to(device)

    mask_generator = SamAutomaticMaskGenerator(sam.lora_sam)
    dataset = SA1B_Dataset("./data", transform=input_transform, target_transform=target_transform)
    image, target = dataset.__getitem__(122)
    image = TF.resize(image, size=[160, 256])
    target = TF.resize(target, size=[160, 256], interpolation=TF.InterpolationMode.NEAREST)
    if high_res:
        image = TF.resize(image, size=[800, 1280])
        target = TF.resize(target, size=[800, 1280], interpolation=TF.InterpolationMode.NEAREST)

    image_bgr = image[[2, 1, 0], :, :]

    image_bgr_np = image_bgr.permute(1, 2, 0).numpy()  # [C,H,W] -> [H,W,C]
    image_bgr_np = (image_bgr_np * 255).astype(np.uint8)

    c,h,w = image.shape
    resolutions = {
        'Original': (w, h),
        # 'Low-res': (w//4, h//4)
    }

    annotated_images = []

    for res_name, res_size in resolutions.items():
        resized_image = cv2.resize(image_bgr_np, res_size)
        sam_result = mask_generator.generate(resized_image)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(sam_result=sam_result)
        annotated_image = mask_annotator.annotate(scene=resized_image.copy(), detections=detections)
        annotated_images.append((res_name, annotated_image))

    images_to_plot = [image_bgr_np] + [img for _, img in annotated_images]
    titles = ['Original'] + [name for name, _ in annotated_images]

    sv.plot_images_grid(
        images=images_to_plot,
        grid_size=(1, len(images_to_plot)),
        titles=titles
    )

if __name__ == "__main__":
    # inference the original sam with point prompt
    orig_path = ".\checkpoints\sam_vit_b_01ec64.pth"
    tuned_path = ".\logs\checkpoints\MyFastSAM-epoch=08-val_loss=1.1954.ckpt"
    
    inference(checkpoint=orig_path, origin=True)
    # inference the lora sam with point prompt
    inference(checkpoint=tuned_path)

    # inference the original sam for the entire image
    inference_all(checkpoint=orig_path, origin=True)
    # inference the lora sam for the entire image
    inference_all(checkpoint=tuned_path)
