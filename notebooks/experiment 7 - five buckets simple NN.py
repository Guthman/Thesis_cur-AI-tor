import ujson as json
import pandas as pd
import pytorch_lightning as pl
import requests
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger('lightning_logs', name='Cur-AI-tor - engineered')

on_colab = False
# on_colab = True

# Hyperparameters
num_epochs = 100
learning_rate = 1e-4
l2_norm = 0.01

# Trainer parameters
gpus = 0
num_sanity_val_steps = 0
num_processes = 1
pl.seed_everything(3)


# Helper function to download files
def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Load x data
if on_colab:
    download_file('https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frwwzrj6ghal/b/thesis/o/micro_dataset1_resnet18_output_identity.json')
    data_dir = r'micro_dataset1_resnet18_output_identity.json'
else:
    data_dir = r'F:\temp\thesisdata\micro_dataset_1\micro_dataset1_resnet18_output_identity.json'

with open(data_dir, 'r') as f:
    data_dict_list = json.load(f)

data_dict = {}
for element in data_dict_list:
    data_dict.update(element)

# Show first two elements of the dict
# dict(itertools.islice(data_dict.items(), 2))
df_x = pd.DataFrame.from_dict(data_dict, orient='index')
df_x.head()

# Load y data
if on_colab:
    download_file('https://objectstorage.eu-frankfurt-1.oraclecloud.com/n/frwwzrj6ghal/b/thesis/o/SAATCHI_MICRO_DATASET_PRICE_VIEWSLIKES.tsv')
    data_dir = 'SAATCHI_MICRO_DATASET_PRICE_VIEWSLIKES.tsv'
else:
    data_dir = r'F:\temp\thesisdata\SAATCHI_MICRO_DATASET_PRICE_VIEWSLIKES.tsv'

df_y = pd.read_csv(data_dir, sep='\t')
df_y.set_index('FILENAME', inplace=True)
# df_y['PRICE_BIN'] = pd.qcut(df_y['PRICE'], q=5)
df_y['PRICE_BIN_IDX'] = pd.qcut(df_y['PRICE'], q=5, labels=[0, 1, 2, 3, 4])
# df_y['LIKES_VIEWS_RATIO_BIN'] = pd.qcut(df_y['LIKES_VIEWS_RATIO'], q=5)
df_y['LIKES_VIEWS_RATIO_BIN_IDX'] = pd.qcut(df_y['LIKES_VIEWS_RATIO'], q=5, labels=[0, 1, 2, 3, 4])
df_y = df_y.astype({'PRICE_BIN_IDX': int, 'LIKES_VIEWS_RATIO_BIN_IDX': int})
df_y.drop(['PRICE', 'LIKES_VIEWS_RATIO'], axis=1, inplace=True)

df_y.head()

# Join x and y into a single dataframe
df = df_y.join(df_x)
df.head()


class SaatchiDataset(Dataset):
    training_set = df[:13000]
    validation_set = df[13000:14000]
    test_set = df[14000:]

    @property
    def targets(self):
        return self.targets_

    @property
    def data(self):
        return self.data_

    def __init__(self, stage: str = None, target_selection=None):
        self.stage = stage
        self.target_selection = target_selection

        if self.stage == 'train':
            self.dataset = self.training_set
        elif self.stage == 'validation':
            self.dataset = self.validation_set
        elif self.stage == 'test':
            self.dataset = self.test_set
        else:
            print(f'Invalid stage specified: "{stage}" , valid options are: [train, validation, test].')
            self.dataset = None

        self.data_ = self.dataset.drop(['PRICE_BIN_IDX', 'LIKES_VIEWS_RATIO_BIN_IDX'], axis=1).values

        if self.target_selection == 'price':
            self.targets_ = self.dataset['PRICE_BIN_IDX'].values
        elif self.target_selection == 'likes_view_ratio':
            self.targets_ = self.dataset['LIKES_VIEWS_RATIO_BIN_IDX'].values
        else:
            print(
                f'Invalid target selection specified: "{target_selection}"'
                f', valid options are: [price, likes_view_ratio].')

    def __getitem__(self, index):
        return torch.as_tensor(self.data_[index]).float(), torch.as_tensor(self.targets_[index]).long()

    def __len__(self):
        return len(self.data_)


class SaatchiDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 target_selection: str = 'price'):
        super().__init__()
        self.batch_size = batch_size
        self.data = None
        self.num_workers = num_workers
        self.target_selection = target_selection

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.data = SaatchiDataset(stage='train', target_selection=self.target_selection)
        else:
            self.data = SaatchiDataset(stage=stage, target_selection=self.target_selection)

    def train_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers)


class SaatchiMLP(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.hparams.l2_norm = l2_norm
        self.hparams.lr = learning_rate

        self.layers = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Linear(10, 5)

        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)

        # Logic for calculating and printing accuracy
        step_counter.increment()
        if step_counter.step_count % 50 == 0:

            pred = np.array([x.argmax() for x in y_hat.detach().numpy()])
            y_ = y.detach().numpy()
            correct_preds = np.sum(y_ == pred)
            acc = round(correct_preds / y_.shape[0], 1)
            # print(f'y_hat = {y_hat}; y = {y}')
            print(f'Accuracy at step {step_counter.step_count}: {acc}')

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.layers(x)
        # print(f'y = {y}; y_hat = {y_hat}')
        loss = self.ce(y_hat, y)
        self.log('validation_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.l2_norm)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  factor=0.2,
                                                                  patience=2,
                                                                  min_lr=1e-6,
                                                                  verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'validation_loss',
            'reduce_on_plateau': True
        }

        return [optimizer], [scheduler]


class StepCounter(object):
    def __init__(self):
        self.step_count = 0

    def increment(self):
        self.step_count = self.step_count + 1

    @property
    def get_step_count(self):
        return self.step_count

step_counter = StepCounter()

saatchi_data = SaatchiDataModule(target_selection='price',
                                 batch_size=128,
                                 num_workers=1)

num_sanity_val_steps = 0
num_processes = 1

saatchi_mlp = SaatchiMLP()

trainer = pl.Trainer(auto_scale_batch_size='power',
                     gpus=0,
                     deterministic=True,
                     max_epochs=5,
                     num_sanity_val_steps=num_sanity_val_steps,
                     num_processes=num_processes)

# trainer.fit(saatchi_mlp, saatchi_data)


def train():
    trainer.fit(saatchi_mlp, saatchi_data)


if __name__ == '__main__':
    train()
