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

def old_train(model: Sam, train_loader, test_loader, opt, summary_writer: SummaryWriter):
    # this is data + forward
    model.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=opt.lr)

    # check existing data
    writer_dir = os.path.join(opt.save_dir, opt.expname)
    start_epoch = 0
    global_step = 0

    if start_epoch == opt.num_epochs - 1:
        return
    
    for epoch in range(start_epoch, opt.num_epochs): 
        model.train()
        for batch_idx, (image, target) in tqdm(enumerate(train_loader)):
            batched_input, target = construct_batched_input(image, target)
            # batched_input: a list (len=batch_size) of dics: 'img':3,160,256, 'boxes':16,4
            # batched_targets: a list (len=batch_size) of (16,160,256)
            target = torch.stack(target, dim=0).cuda()
            images = torch.stack([i["image"] for i in batched_input], dim=0).cuda()
            # boxes = torch.stack([i["boxes"] for i in batched_input], dim=0).cuda()

            # target:8, 16, 160, 256
            # boxes:8, 16, 4
            # images:8, 3, 160, 256
            h, w = target.shape[-2:]
            pred = model.forward(batched_input, multimask_output=False)
            pred = torch.stack([i["masks"] for i in pred], dim=0).cuda()    #8,16,1,160,256
            pred = pred.reshape(-1, 1, h, w)

            focal_loss, dice_loss = calc_loss(pred, target)
            loss = focal_loss + 0.01 * dice_loss

            with torch.no_grad():
                binary_mask = pred > model.mask_threshold
                binary_mask = binary_mask[:, 0, :, :]
                target_mask = target.reshape(-1, h, w)

                intersect = (binary_mask * target_mask).sum(dim=(-1, -2))
                union = binary_mask.sum(dim=(-1, -2)) + target_mask.sum(dim=(-1, -2)) - intersect
                ious = intersect.div(union)
                train_ious = torch.mean(ious)

            if batch_idx % 50 == 0:
                print("ITER [{}] / EPOCH {}, loss: {}, focal: {}, dice: {}".format(global_step, epoch, loss.item(), focal_loss.item(), dice_loss.item()))

                # visualize the first image
                masks = [binary_mask[:10].cpu().numpy(), target_mask[:10].cpu().numpy()]
                image_cpu = images[0].cpu().permute(1, 2, 0)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                for axis, mask in zip(axes, masks):
                    axis.imshow(image_cpu / 255.)
                    for m in mask:
                        show_mask(m, axis, random_color=True)

                summary_writer.add_figure("train_{}".format(batch_idx), fig, epoch)
            
            summary_writer.add_scalar("train/loss", loss.item(), global_step)
            summary_writer.add_scalar("train/focal", focal_loss.item(), global_step)
            summary_writer.add_scalar("train/dice", dice_loss.item(), global_step)
            summary_writer.add_scalar("train/mIoU", train_ious.item(), global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        if epoch % 2 == 0 and epoch > 1:
            name = os.path.join(writer_dir, "LoRA_sam_{}.pt".format(epoch))
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step
            }

            torch.save(ckpt, name)
    
    # Save Model at the end
    name = os.path.join(writer_dir, "LoRA_sam_{}.pt".format(epoch))
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step
    }
    torch.save(ckpt, name)

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
        # use_small_subset=960
    )
    print(len(train_loader))

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