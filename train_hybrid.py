from tools.trainers.endodepth import plEndoDepth
from tools.options.endodepth import EndoDepthOptions
from torch.utils.data import DataLoader
import pytorch_lightning as pl
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

    print("\n" + "="*80)
    print("HYBRID CNN-TRANSFORMER DEPTH ESTIMATION TRAINING")
    print("="*80)
    print(f"Architecture: {opt.backbone.upper()}")
    if opt.backbone == 'transformer':
        print(f"Encoder: Swin Transformer {opt.encoder}")
    print(f"Multi-scale: {opt.scales}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Learning rate: {opt.learning_rate}")
    print(f"Epochs: {opt.num_epochs}")
    print(f"Normal loss: {opt.use_normal_loss}")
    print("="*80 + "\n")

    model = plEndoDepth(options=opt, verbose=2)

    train_loader = DataLoader(
        model.train_set, opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=True if opt.num_workers > 0 else False
    )

    logger = TensorBoardLogger(save_dir='.', name='hybrid_logs')
    checkpoint = ModelCheckpoint(
        monitor="train_loss",
        save_top_k=3,
        mode="min",
        filename="hybrid-{epoch:02d}-{train_loss:.4f}"
    )
    
    trainer = pl.Trainer(
        max_epochs=opt.num_epochs,
        precision=32,
        callbacks=[checkpoint],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    print("Starting training...")
    trainer.fit(model, train_loader)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print(f"Best model saved in: {checkpoint.best_model_path}")
    print("="*80)
