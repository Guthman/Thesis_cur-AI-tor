import ujson as json
import pandas as pd
from sklearn.preprocessing import RobustScaler

import pytorch_lightning as pl
from pl_bolts.models.regression import LinearRegression

import torch
from torch.utils.data import Dataset, DataLoader

on_colab = False
# on_colab = True


# Load x data
if on_colab:
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
    data_dir = 'SAATCHI_MICRO_DATASET_PRICE_VIEWSLIKES.tsv'
else:
    data_dir = r'F:\temp\thesisdata\SAATCHI_MICRO_DATASET_PRICE_VIEWSLIKES.tsv'

df_y = pd.read_csv(data_dir, sep='\t')
df_y.set_index('FILENAME', inplace=True)

# scale y data
scaler_price = RobustScaler().fit(df_y[['PRICE']].values)
scaler_rating = RobustScaler().fit(df_y[['LIKES_VIEWS_RATIO']].values)

scalar_params_price = scaler_price.get_params(deep=True)
scalar_params_rating = scaler_rating.get_params(deep=True)

scaled_price = scaler_price.transform(df_y[['PRICE']].values)
scaled_rating = scaler_rating.transform(df_y[['LIKES_VIEWS_RATIO']].values)
df_y['PRICE'] = scaled_price
df_y['LIKES_VIEWS_RATIO'] = scaled_rating
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

    def __init__(self, transform=None, stage: str = None, target_selection=None):
        self.stage = stage
        self.target_selection = target_selection

        if self.stage == 'train':
            dataset = self.training_set
        elif self.stage == 'validation':
            dataset = self.validation_set
        elif self.stage == 'test':
            dataset = self.test_set
        else:
            print(f'Invalid stage specified: "{stage}" , valid options are: [train, validation, test].')

        self.data_ = dataset.drop(['PRICE', 'LIKES_VIEWS_RATIO'], axis=1).values

        if self.target_selection == 'price':
            self.targets_ = dataset['PRICE'].values
        elif self.target_selection == 'likes_view_ratio':
            self.targets_ = dataset['LIKES_VIEWS_RATIO'].values
        else:
            print(
                f'Invalid target selection specified: "{target_selection}" , valid options are: [price, likes_view_ratio].')

    def __getitem__(self, index):
        return torch.as_tensor(self.data_[index]), torch.as_tensor(self.targets_[index]).unsqueeze(-1)

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

    def setup(self, stage):
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


saatchi_data = SaatchiDataModule(target_selection='price', batch_size=128, num_workers=4)
input_dim = 512
num_sanity_val_steps = 2
num_processes = 1

model = LinearRegression(input_dim=input_dim)
trainer = pl.Trainer(num_sanity_val_steps=num_sanity_val_steps,
                     num_processes=num_processes)


def train():
    trainer.fit(model.double(), saatchi_data)


if __name__ == '__main__':
    train()
