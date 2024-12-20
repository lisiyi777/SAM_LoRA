# SAM on Low-Resolution Images Using LoRA

The Segmentation Anything Model (SAM) demonstrates impressive generalization and real-time interactivity in image segmentation. However, its performance significantly degrades on low-resolution datasets. To address these limitations, we propose fine-tuning SAM with Low-Rank Adaptation (LoRA), enabling efficient adaptation to low-resolution inputs. Our approach incorporates tailored loss functions, optimized prompt generation strategies, and a scalable inference framework to evaluate performance across object scales. Validated on CelebAMask-HQ, our method achieves superior accuracy and significantly reduces annotation time, demonstrating its effectiveness for low-resolution segmentation in resource-constrained settings.

# Usage
1. Pre-process and save masks.
  ```
  python dataset.py
  ```
3. Train LoRA-SAM
  ```
  python train.py
  ```
4. Inference(revise the checkpoint path)
  ```
  python inference.py
  ```
5. ncut visualization
  ```
  # for original SAM
  pythoon ncut_vis_origin_sam.py

  # for LoRA-SAM
  python tuned_sam_ncut.py
  ```
