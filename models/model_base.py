import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn, optim
import argparse

from torch.utils.data import SubsetRandomSampler, DataLoader

from utils.dataset_one_picture_to_grid_europe import DatasetGridOverEurope


class GMAN_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--lr_patience', type=int, default=4)
        parser.add_argument('--es_patience', type=int, default=30)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.validation_step_outputs = []

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=self.hparams.lr_patience),
            'monitor': 'val_loss',  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        # reduction="sum" means the loss is calculated for the whole image
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        self.log("val_loss", loss, prog_bar=True)

    def test_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["test_loss"]
        avg_loss /= len(outputs)
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": {"test_loss": avg_loss}}

    def test_step(self, batch, batch_idx):
        """Calculate the loss (MSE per default) on the test set normalized and denormalized."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        factor = 47.83
        loss_denorm = self.loss_func(y_pred.squeeze() * factor, y * factor)
        self.log("MSE", loss)
        self.log("MSE_denormalized", loss_denorm)

    def _epoch_start(self):
        print('\n')


class Precipitation_base(GMAN_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = GMAN_base.add_model_specific_args(parent_parser)

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=6)
        parser.add_argument("--num_output_images", type=int, default=6)
        parser.add_argument("--kernels_per_layer", type=int, default=2)
        parser.add_argument("--cell_cutoff", type=float, default=16)
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)
        parser.add_argument("--val_check_interval", type=float, default=None)

        return parser

    def __init__(self, hparams):
        super(Precipitation_base, self).__init__(hparams=hparams)
        self.europe_dataset = None

        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        self.europe_dataset = DatasetGridOverEurope(
            dataset_path=self.hparams.dataset_folder,
            time_steps=self.hparams.num_input_images,
            future_look=self.hparams.num_output_images,
            cell_path=self.hparams.cell_path,
            cell_cutoff=self.hparams.cell_cutoff,
            fast_dev_run=self.hparams.fast_dev_run,
        )

        self.europe_dataset.count_data_rows()
        self.europe_dataset.load_grid()
        self.europe_dataset.normalize()

        num_train = len(self.europe_dataset)  # Number of training samples (total - past and future look)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))

        np.random.seed(123)
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        return DataLoader(self.europe_dataset, batch_size=self.hparams.batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.europe_dataset, batch_size=self.hparams.batch_size, sampler=self.valid_sampler)


def get_test_dataset(dataset_path, past_look, future_look, B=16, fast_dev_run=False, cell_path='', cell_cutoff=100):
    europe_dataset = DatasetGridOverEurope(
        dataset_path=dataset_path,
        time_steps=past_look,
        future_look=future_look,
        fast_dev_run=fast_dev_run,
        cell_cutoff=cell_cutoff,
        cell_path=cell_path,
    )

    europe_dataset.count_data_rows()
    europe_dataset.load_grid()
    europe_dataset.normalize()

    data_loader = DataLoader(europe_dataset, batch_size=B, shuffle=False)

    return data_loader
