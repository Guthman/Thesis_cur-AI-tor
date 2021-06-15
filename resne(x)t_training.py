import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from saatchi_datamodules import SaatchiImageDataModule
import logging


class ResNext(pl.LightningModule):
    def __init__(self,
                 num_classes=5):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.l2_norm = l2_norm
        self.hparams.lr = learning_rate
        self.ce = nn.CrossEntropyLoss()

        # Define model
        self.model = models.resnext50_32x4d(pretrained=True)
        # self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

        # Define extra metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.validation_accuracy = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        preds = self.model(x)
        preds = F.softmax(preds, dim=1)
        loss = self.ce(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value

        # Calculate and log accuracy
        self.log('train_acc', accuracy(preds, y))
        # self.train_accuracy(preds, y)
        # self.log('training_accuracy', self.train_accuracy, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        preds = self.model(x)
        preds = F.softmax(preds, dim=1)
        loss = self.ce(preds, y)
        self.log('validation_loss', loss)  # lightning detaches your loss graph and uses its value

        # Calculate and log accuracy
        self.log('validation_acc', accuracy(preds, y))
        # self.validation_accuracy(preds, y)
        # self.log('validation_accuracy', self.validation_accuracy, on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.l2_norm)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  factor=lr_scheduler_factor,
                                                                  patience=lr_scheduler_patience,
                                                                  min_lr=lr_scheduler_min_lr,
                                                                  verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'validation_loss',
            'reduce_on_plateau': True
        }

        return [optimizer], [scheduler]


# Trainer parameters
gpus = 4
num_sanity_val_steps = 2
num_processes = 12
pl.seed_everything(3)
num_workers = 12
batch_size = 1024
deterministic = False
image_size = 128
limit_dataset_size_to = 100000
persistent_workers = True
pin_memory = True

# Hyperparameters
num_epochs = 3
learning_rate = 2e-4
l2_norm = 0.0
lr_scheduler_factor = 0.2
lr_scheduler_patience = 16
lr_scheduler_min_lr = 1e-11
gradient_clip_val = 2.5
stochastic_weight_avg = True

pl_logger = logging.getLogger("lightning")
pl_logger.propagate = False

model = ResNext()

saatchi_images = SaatchiImageDataModule(
    path_to_image_tar='/home/rguthman/data/micro_dataset1.tar',
    image_extraction_path='./nu',
    data_format='folder',
    path_to_target_data='/home/rguthman/cur/saatchi_targets.csv',
    batch_size=batch_size,
    image_size=image_size,
    limit_dataset_size_to=limit_dataset_size_to,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory)

trainer = pl.Trainer(gpus=gpus,
                     deterministic=deterministic,
                     max_epochs=num_epochs,
                     num_sanity_val_steps=num_sanity_val_steps,
                     num_processes=num_processes,
                     gradient_clip_val=gradient_clip_val,
                     gradient_clip_algorithm='norm',
                     stochastic_weight_avg=stochastic_weight_avg,
                     precision=16,
                     default_root_dir='/home/rguthman/tmp/cur_output',
                     accelerator="ddp_spawn"
                     )


def train():
    trainer.fit(model, saatchi_images)

if __name__ == '__main__':
    train()
