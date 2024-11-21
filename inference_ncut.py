import cv2
from plotter import Plotter
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from dataset import *
from model import *
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def inference(checkpoint = "sam_vit_b_01ec64.pth"):
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
        input_point = np.array([[500,200],[500,200]])
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
        input_point = np.array([[100,40],[100,40]])
        input_label = np.array([0,1])

    plotter = Plotter(checkpoint, high_res=high_res)
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
    
    plotter.inference_plot(image, target, **kwargs)

def inference_all(checkpoint = "sam_vit_b_01ec64.pth"):
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

    kwargs = {"lora_rank": 4, "lora_scale": 1, "high_res":high_res}
    lorasam = LoRASAM(**kwargs).to(device)
    mask_generator = SamAutomaticMaskGenerator(lorasam.sam)
    dataset = SA1B_Dataset("./data", transform=input_transform, target_transform=target_transform)
    image, target = dataset.__getitem__(66)
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
    # inference()
    inference_all()