import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def calc_IoU(pred, target):
    # pred = torch.from_numpy(pred).to(target.device).bool()
    pred = pred.to(target.device).bool()
    target = target.bool()

    intersections = torch.sum(pred & target, dim=(1, 2))
    unions = torch.sum(pred | target, dim=(1, 2))        
    epsilon = 1e-7
    ious = intersections.float() / (unions.float() + epsilon)        
    return ious

def calculateIoU(pred, gt):
    intersect = (pred * gt).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) - intersect
    ious = intersect.div(union)
    return ious

def get_bbox_from_mask(target):
    bbox = []
    valid_index = []
    for idx, mask in enumerate(target):
        if mask.sum() > 0.:
            coord_y, coord_x = torch.where(mask > 0)
            if len(coord_x) == 0:
                import pdb; pdb.set_trace()
            x1, y1 = coord_x.min(), coord_y.min()
            x2, y2 = coord_x.max(), coord_y.max()
            bbox.append(torch.stack([x1, y1, x2, y2])) # (Image Coordinate)
            valid_index.append(idx)

    return torch.stack(bbox), valid_index

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

def show_image(image, target, row=12, col=12):
    # image: numpy image
    # target: mask [N, H, W]
    bbox, vidx = get_bbox_from_mask(target)
    target = target[vidx]
    fig, axs = plt.subplots(row, col, figsize=(20, 12))
    for i in range(row):
        for j in range(col):
            if i*row+j < target.shape[0]:
                box = bbox[i*row+j]
                canvas = image.copy()
                cv2.rectangle(canvas, (box[0].item(), box[1].item()),
                                (box[2].item(), box[3].item()), (255, 0, 0))
                axs[i, j].imshow(canvas)
                axs[i, j].imshow(target[i*row+j], alpha=0.5)
            else:
                axs[i, j].imshow(image)
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("image.png")