import torch
import numpy as np
from model import LoRASAM
import matplotlib.pyplot as plt
from segment_anything import SamPredictor
import torchvision.transforms as transforms

class Plotter:
    def __init__(self, checkpoint="sam_vit_b_01ec64.pth", high_res=False):
        self.device = "cuda"

        kwargs = {"lora_rank": 4, "lora_scale": 1, "high_res":high_res}
        self.sam = LoRASAM(**kwargs).to(self.device)
        self.predictor = SamPredictor(self.sam.sam)
        self.reverse = transforms.ToPILImage()

    def __to_numpy(self, img):
        return np.array(self.reverse(img))

    def inference_plot(self, img, target, coords=None, labels=None, bboxes=None):
        img = self.__to_numpy(img)
        self.predictor.set_image(img)

        masks, scores, logits = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=None if bboxes is None else bboxes[None, :],
            multimask_output=True,
        )

        if bboxes is not None:
            gt_masks = self.sam.box_sample(bboxes, target)
        else:
            gt_masks = self.sam.point_sample(coords, labels, target)

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
