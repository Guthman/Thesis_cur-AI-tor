import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import as_tensor
import tarfile
from glob import glob
from pathlib import Path


class SaatchiImageDataset(Dataset):
    def untar_images(self,
                     path,
                     image_extraction_path):
        self.dataset_tarfile = tarfile.open(path)
        self.dataset_tarfile.extractall(image_extraction_path)
        self.dataset_tarfile.close()
        return glob(image_extraction_path + '/*/*')

    def __init__(self,
                 stage,
                 target_selection,
                 source_selection,
                 path_to_image_tar,
                 image_extraction_path,
                 path_to_target_data,
                 image_size,
                 limit_dataset_size_to
                 ):
        if source_selection != 'images':
            print('This DataModule only supports images!')
        self.path_to_image_tar = path_to_image_tar
        self.image_extraction_path = image_extraction_path
        self.path_to_target_data = path_to_target_data
        self.dataset_tarfile = None
        self.filelist = None
        self.image_size = (image_size, image_size)
        self.split_fractions = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
        self.limit_dataset_size_to = limit_dataset_size_to

        # Load target data
        self.targets_df = pd.read_csv(path_to_target_data, header=None)
        self.targets_df.columns = ['FILENAME', 'PRICE', 'LIKES_VIEWS_RATIO']
        # Bin the values
        self.targets_df['PRICE_BIN_IDX'] = pd.qcut(self.targets_df['PRICE'], q=5, labels=[0, 1, 2, 3, 4])
        self.targets_df['LIKES_VIEWS_RATIO_BIN_IDX'] = pd.qcut(self.targets_df['LIKES_VIEWS_RATIO'],
                                                               q=5, labels=[0, 1, 2, 3, 4])
        self.targets_df = self.targets_df.astype({'PRICE_BIN_IDX': int, 'LIKES_VIEWS_RATIO_BIN_IDX': int})
        self.targets_df.drop(['PRICE', 'LIKES_VIEWS_RATIO'], axis=1, inplace=True)
        self.targets_df.set_index('FILENAME', inplace=True)
        # Remove any duplicates
        self.targets_df = pd.DataFrame(
            self.targets_df.reset_index().drop_duplicates(subset=['FILENAME'])).set_index('FILENAME')

        # Validate arguments
        if stage not in ['train', 'validation', 'test']:
            print(f'Invalid stage specified: "{stage}" , valid options are: [train, validation, test].')
        else:
            self.stage = stage

        if target_selection not in ['PRICE_BIN_IDX', 'LIKES_VIEWS_RATIO_BIN_IDX']:
            print(
                f'Invalid target selection specified: "{target_selection}"'
                f', valid options are: [PRICE_BIN_IDX, LIKES_VIEWS_RATIO_BIN_IDX].')
        else:
            self.target_selection = target_selection

        # Define transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # Create dataset
        if Path(image_extraction_path).exists():
            print('Image extraction path already exists, using existing contents!')
            self.filelist = glob(image_extraction_path + '/*/*')
        else:
            # Only extract if folder doesn't exist yet
            self.filelist = self.untar_images(self.path_to_image_tar, self.image_extraction_path)

        if self.limit_dataset_size_to is not None:
            self.filelist = self.filelist[:self.limit_dataset_size_to]

        # Calculate how many images belong in train, validation, and test
        self.train_fraction = int(len(self.filelist) * self.split_fractions['train'])
        self.validation_fraction = int(len(self.filelist) * self.split_fractions['validation'])
        self.test_fraction = int(len(self.filelist) * self.split_fractions['test'])

        if self.stage == 'train':
            self.data_ = self.filelist[:self.train_fraction]
        elif self.stage == 'validation':
            self.data_ = self.filelist[self.train_fraction:self.train_fraction + self.validation_fraction]
        elif self.stage == 'test':
            self.data_ = self.filelist[self.train_fraction + self.validation_fraction:]

    def __getitem__(self, index):
        path = self.data_[index]
        img = Image.open(path)
        # Resize image if it's not the requested size TODO: warning or solution for when image is not square
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.ANTIALIAS)
        image_tensor = self.transform(img)
        filename = Path(path).name
        target = self.targets_df.loc[filename][self.target_selection]
        return image_tensor, as_tensor(target)

    def __len__(self):
        return len(self.data_)


class SaatchiImageDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 128,
                 num_workers: int = 4,
                 target_selection: str = 'LIKES_VIEWS_RATIO_BIN_IDX',
                 source_selection: str = 'images',
                 path_to_image_tar: str = './data/saatchi512.tar',
                 image_extraction_path: str = './data/images/',
                 path_to_target_data: str = './data/saatchi_targets.csv',
                 image_size: int = 512,
                 limit_dataset_size_to: int = None
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.data = None
        self.num_workers = num_workers
        self.target_selection = target_selection
        self.source_selection = source_selection
        self.path_to_image_tar = path_to_image_tar
        self.image_extraction_path = image_extraction_path
        self.path_to_target_data = path_to_target_data
        self.image_size = image_size
        self.limit_dataset_size_to = limit_dataset_size_to

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == 'fit':
            self.data = SaatchiImageDataset(stage='train',
                                            target_selection=self.target_selection,
                                            source_selection=self.source_selection,
                                            image_extraction_path=self.image_extraction_path,
                                            path_to_target_data=self.path_to_target_data,
                                            path_to_image_tar=self.path_to_image_tar,
                                            image_size=self.image_size,
                                            limit_dataset_size_to=self.limit_dataset_size_to)
        else:
            self.data = SaatchiImageDataset(stage=stage,
                                            target_selection=self.target_selection,
                                            source_selection=self.source_selection,
                                            image_extraction_path=self.image_extraction_path,
                                            path_to_target_data=self.path_to_target_data,
                                            path_to_image_tar=self.path_to_image_tar,
                                            image_size=self.image_size,
                                            limit_dataset_size_to=self.limit_dataset_size_to)

    def train_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=True
                          )

    def val_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=True
                          )

    def test_dataloader(self):
        return DataLoader(self.data,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=True
                          )
