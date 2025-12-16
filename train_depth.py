from tools.trainers.endodepth import plEndoDepth
from tools.options.endodepth import EndoDepthOptions
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import json
import torch

torch.set_float32_matmul_precision('medium')

options = EndoDepthOptions()
options.parser.add_argument('--config', type=str, help='path to config json', required=True)

if __name__ == "__main__":

    opt = options.parse()
    args_dict = vars(opt)
    with open(opt.config, 'r') as config_file:
        args_dict.update(json.load(config_file))

    model = plEndoDepth(options=opt, verbose=2)

    train_loader = DataLoader(
        model.train_set, opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True if opt.num_workers > 0 else False
    )

    logger = TensorBoardLogger(save_dir='.')
    checkpoint = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=1,
        mode="min",
        filename="endodepth-{epoch:02d}-{train_loss:.4f}"
    )

    trainer = pl.Trainer(
        max_epochs=opt.num_epochs,
        precision=32,
        callbacks=[checkpoint],
        logger=logger,
        gradient_clip_val=1.0
    )
    trainer.fit(model, train_loader)

