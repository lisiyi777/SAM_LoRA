import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from segment_anything.modeling.sam import Sam
import argparse
from torch.utils.tensorboard import SummaryWriter
from model import MyFastSAM
from dataset import get_loaders


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    model = MyFastSAM(
        lora_rank=args.rank,
        lora_scale=args.scale,
        lr=args.lr,
        linear=args.linear,
        conv2d=args.conv2d,
        convtrans2d = args.convtrans2d,
        num_epochs=args.num_epochs
    )

    logger = TensorBoardLogger(save_dir=args.save_dir, name="MyFastSAM")

    os.makedirs("./checkpoints", exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",  # Metric to monitor
        mode="max",  # Save the checkpoint with the lowest validation loss
        dirpath="./checkpoints",
        filename="MyFastSAM-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,  # Save the top 3 best models
        save_last=True,
    )

    train_loader, val_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_small_subset=960
    )
    print(len(train_loader))
    print(len(val_loader))

    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",  
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./logs")
    parser.add_argument("--expname", type=str, default="LoRA_SAM")
    parser.add_argument("--linear", type=bool, default=True)
    parser.add_argument("--conv2d", type=bool, default=False)
    parser.add_argument("--convtrans2d", type=bool, default=False)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default='../SAM_LoRA/data')

    args = parser.parse_args()

    train(args)