import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam

"""
The EarlyStopping callback can be used to monitor a validation metric and stop the training when no improvement is observed.

"""


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    """
    You can set your own model specific arguments here in a static
    method. Pytorch will sort them out.
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group()
        parser.add_argument("--some-argument", type=str, default="reasonable-default")
        return parent_parser

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        # If you have multiple optimizers
        # just return them one after the other
        return Adam(self.parameters(), lr=1e-3)

    """
    The training step automatically handles the optimizer
    step for you. You can do manual optimization but this
    should handle most cases.
    """

    def training_step(self, batch, idx):
        x, y = batch

        logits = self(x)

        # util function from Torch that gives
        # negative loss likelihood
        loss = F.nll_loss(logits, y)

        # loss is returned so that you can optimise the
        # weights with respect to this loss

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)

        """
        in lightning just 'log' this and
        then the early callbacks can automate model
        stopping
        """
        self.log("val_loss", loss, on_step=True)
