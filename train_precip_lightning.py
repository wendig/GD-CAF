import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import loggers
from models.GD_CAF import GDCAF
from models.model_base import Precipitation_base
from models.unets import UNetDS_Attention


def train_regression(hparams):
    net = None
    if hparams.model == "GDCAF":
        net = GDCAF(hparams=hparams)
    elif hparams.model == "UNetDS_Attention":
        net = UNetDS_Attention(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    naming = f"_L_{hparams.L}_K_{hparams.K}_kpl_{hparams.kernels_per_layer}_in_{hparams.num_input_images}_out_{hparams.num_output_images}_cut_{hparams.cell_cutoff}"
    print(f"Naming: {naming}")

    tb_logger = loggers.TensorBoardLogger(save_dir=hparams.default_save_path, name=net.__class__.__name__)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.default_save_path + '/' + net.__class__.__name__,
        filename=net.__class__.__name__ + naming + "_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_logger = LearningRateMonitor()

    earlystopping_callback = EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=hparams.es_patience) # is effectively half (due to a bug in pytorch-lightning)

    trainer = pl.Trainer(fast_dev_run=hparams.fast_dev_run,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1,
                         max_epochs=hparams.epochs,
                         default_root_dir=hparams.default_save_path,
                         logger=tb_logger,
                         callbacks=[lr_logger, checkpoint_callback, earlystopping_callback],
                         val_check_interval=hparams.val_check_interval,
                         enable_checkpointing=True,
                         benchmark=True,
                         )
    trainer.fit(net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Precipitation_base.add_model_specific_args(parser)

    parser.add_argument('--dataset_folder', default='data/train', type=str)
    parser.add_argument('--default_save_path', default='db/trained_models', type=str)
    parser.add_argument('--cell_path', default='utils/cell_positions.csv', type=str)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args(args=[])
    # General
    args.fast_dev_run = False
    args.model = "GDCAF"
    args.kernels_per_layer = 2  # Number of attention mechanism
    # GDCAF
    args.L = 2  # Number of attention block
    args.K = 4  # Number of attention mechanism
    args.poll_input = True
    args.poll_qk = True
    args.poll_v = True
    ###
    args.cell_cutoff = 16
    ###
    args.num_input_images = 6
    args.num_output_images = 6
    # SmaAtUNet
    args.bilinear = False
    args.reduction_ratio = 16
    args.n_classes = args.cell_cutoff
    args.n_channels = args.num_input_images * args.cell_cutoff
    # Continue from checkpoint
    # args.resume_from_checkpoint = ""

    train_regression(args)
