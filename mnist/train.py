"""
Training script that should train an MNIST classifier model.
"""
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import LitMNIST
from dataset import MNISTDataModule

"""
Things to note:
  Early Stopping is good
  Model Checkpointing makes it really easy
"""


def train():

    # PROGRAM LEVEL ARGS
    parser = ArgumentParser()
    parser.add_argument("--save_top_k", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./datasets/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--logging", type=str, default="./logging")
    # add model specific args
    parser = LitMNIST.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    # Here later on add a TensorBoardLogger?

    args = parser.parse_args()

    model = LitMNIST()
    # you can save a model based on some kind of metric?

    callbacks = [
        ModelCheckpoint(
            dirpath="./checkpoints",
            mode="max",
            monitor="val_loss",
            save_top_k=args.save_top_k,
        ),
        EarlyStopping(monitor="val_loss", patience=3, mode="max"),
    ]

    logger = TensorBoardLogger(args.logging, name=None)
    # You can add your own callbacks and logging calls as arguments to this.
    trainer = Trainer.from_argparse_args(
        args, callbacks=callbacks, gpus=1, logger=logger
    )

    # use a provided dataset
    dataset = MNISTDataModule(args.batch_size, args.data_dir)

    trainer.fit(model, dataset)


if __name__ == "__main__":
    train()
