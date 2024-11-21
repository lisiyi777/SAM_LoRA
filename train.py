import os
import argparse
from model import MyFastSAM
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataset import get_loaders

def train(args):
    # Set random seed for reproducibility
    seed_everything(42, workers=True)

    # Create logging directory
    os.makedirs(args.save_dir, exist_ok=True)

    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir=args.save_dir, name="MyFastSAM")

    # Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        mode="min",  # Save the checkpoint with the lowest validation loss
        dirpath=os.path.join(args.save_dir, "checkpoints"),
        filename="MyFastSAM-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,  # Save the top 3 best models
    )

    # Early Stopping Callback (optional)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,  # Stop training if no improvement after 5 epochs
        verbose=True,
    )

    # Prepare Dataloaders
    train_loader, val_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_small_subset=True
    )

    # Initialize the model
    model = MyFastSAM(
        lora_rank=args.rank,
        lora_scale=args.scale,
        lr=args.lr,  # Pass learning rate explicitly
        linear=args.linear,
        conv2d=args.conv2d,
        convtrans2d = args.convtrans2d,
        num_epochs=args.num_epochs
    )

    # Trainer
    trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",  # Automatically uses GPU if available
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./logs")
    parser.add_argument("--linear", type=bool, default=True)
    parser.add_argument("--conv2d", type=bool, default=True)
    parser.add_argument("--convtrans2d", type=bool, default=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="./data")

    args = parser.parse_args()

    train(args)
