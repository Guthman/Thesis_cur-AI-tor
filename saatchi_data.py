import pandas as pd
import pytorch_lightning as pl
import ujson as json

class SaatchiDataset(Dataset):
    training_set = df[:88000]
    validation_set = df[88000:93000]
    test_set = df[93000:]

    @property
    def targets(self):
        return self.targets_

    @property
    def data(self):
        return self.data_

    def __init__(self,
                 stage: str = None,
                 target_selection=None,
                 source_selection: str = 'both'):
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
                          num_workers=self.num_workers,
                          pin_memory=True,
                          # multiprocessing_context='spawn'
                          )

    def val_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          # multiprocessing_context='spawn'
                          )

    def test_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          # multiprocessing_context='spawn'
                          )