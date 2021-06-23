import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import as_tensor
import tarfile
from random import shuffle
from glob import glob
from collections import Counter
from pathlib import Path


class SaatchiImageDataset(Dataset):
    def untar_images(self,
                     path,
                     image_extraction_path):
        self.dataset_tarfile = tarfile.open(path)
        self.dataset_tarfile.extractall(image_extraction_path)
        self.dataset_tarfile.close()
        return glob(image_extraction_path + '/**/*.jpg', recursive=True)

    def no_images_per_class(self,
                            filelist: list,
                            targets: pd.DataFrame):
        filelist = [Path(file).name for file in filelist]  # extract filename from full path
        filenames_with_labels = {
            filename: targets.loc[filename][self.target_selection] for filename in filelist}
        class_composition = {f'Class {k}:': v for k, v in dict(Counter(filenames_with_labels.values())).items()}
        return class_composition

    def __init__(self,
                 stage,
                 target_selection,
                 source_selection,
                 data_format,
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
        self.data_format = data_format
        self.dataset_tarfile = None
        self.filelist = None
        self.image_size = (image_size, image_size)
        self.split_fractions = {'train': 0.7, 'validation': 0.15, 'test': 0.15}
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
        self.targets_df.dropna(inplace=True)
        print(f'Target list length: {len(self.targets_df)}')

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
        if self.data_format != 'archive':
            self.filelist = glob(image_extraction_path + '/**/*.jpg', recursive=True)
        else:
            if Path(image_extraction_path).exists():
                print('Image extraction path already exists, using existing contents!')
                self.filelist = glob(image_extraction_path + '/**/*.jpg', recursive=True)
            else:
                # Only extract if folder doesn't exist yet
                self.filelist = self.untar_images(self.path_to_image_tar, self.image_extraction_path)

        # Remove files from filelist if there is no target information
        # Somewhat convoluted because the target list does not contain the path and the filelist does
        self.filenames_with_full_paths = {Path(file).name: file for file in self.filelist}
        self.filename_list = list(set(self.filenames_with_full_paths.keys()).intersection(list(self.targets_df.index)))
        self.filelist = [self.filenames_with_full_paths[file] for file in self.filename_list]
        shuffle(self.filelist)

        if self.limit_dataset_size_to is not None:
            self.filelist = self.filelist[:self.limit_dataset_size_to]

        # Calculate how many images belong in train, validation, and test
        print(f'Total image count: {len(self.filelist)}')
        self.train_fraction = int(len(self.filelist) * self.split_fractions['train'])
        self.validation_fraction = int(len(self.filelist) * self.split_fractions['validation'])
        self.test_fraction = int(len(self.filelist) * self.split_fractions['test'])

        if self.stage == 'train':
            start_position = 0
            end_position = self.train_fraction
            self.data_ = self.filelist[start_position:end_position]
            self.class_composition = self.no_images_per_class(self.data_, self.targets_df)
            print(f'Training set image count: {len(self.data_)}')
            print(f'Images per class in training set: {self.class_composition}')

        elif self.stage == 'validation':
            start_position = self.train_fraction
            end_position = self.train_fraction + self.validation_fraction
            self.data_ = self.filelist[start_position:end_position]
            self.class_composition = self.no_images_per_class(self.data_, self.targets_df)
            print(f'Validation set image count: {len(self.data_)}')
            print(f'Images per class in validation set: {self.class_composition}')

        elif self.stage == 'test':
            start_position = self.train_fraction + self.validation_fraction
            end_position = len(self.filelist) + 1
            self.data_ = self.filelist[start_position:end_position]
            self.class_composition = self.no_images_per_class(self.data_, self.targets_df)
            print(f'Test set image count: {len(self.data_)}')
            print(f'Images per class in test set: {self.class_composition}')

    def __getitem__(self, index):
        path = self.data_[index]
        img = Image.open(path)

        # Resize and/or convert image if it's not in the right format
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.ANTIALIAS)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')

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
                 data_format: str = 'archive',
                 path_to_image_tar: str = '',
                 image_extraction_path: str = './data/images/',
                 path_to_target_data: str = './data/saatchi_targets.csv',
                 image_size: int = 512,
                 limit_dataset_size_to: int = None,
                 pin_memory: bool = False,
                 persistent_workers: bool = False
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.data_ = None
        self.num_workers = num_workers
        self.target_selection = target_selection
        self.source_selection = source_selection
        self.path_to_image_tar = path_to_image_tar
        self.image_extraction_path = image_extraction_path
        self.path_to_target_data = path_to_target_data
        self.image_size = image_size
        self.limit_dataset_size_to = limit_dataset_size_to
        self.data_format = data_format
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage in ('fit', None):
            self.data_ = SaatchiImageDataset(stage='train',
                                             target_selection=self.target_selection,
                                             source_selection=self.source_selection,
                                             image_extraction_path=self.image_extraction_path,
                                             path_to_target_data=self.path_to_target_data,
                                             data_format=self.data_format,
                                             path_to_image_tar=self.path_to_image_tar,
                                             image_size=self.image_size,
                                             limit_dataset_size_to=self.limit_dataset_size_to)
        elif stage in ('validation', None):
            self.data_ = SaatchiImageDataset(stage='validation',
                                             target_selection=self.target_selection,
                                             source_selection=self.source_selection,
                                             image_extraction_path=self.image_extraction_path,
                                             path_to_target_data=self.path_to_target_data,
                                             data_format=self.data_format,
                                             path_to_image_tar=self.path_to_image_tar,
                                             image_size=self.image_size,
                                             limit_dataset_size_to=self.limit_dataset_size_to)
        elif stage in ('test', None):
            self.data_ = SaatchiImageDataset(stage='test',
                                             target_selection=self.target_selection,
                                             source_selection=self.source_selection,
                                             image_extraction_path=self.image_extraction_path,
                                             path_to_target_data=self.path_to_target_data,
                                             data_format=self.data_format,
                                             path_to_image_tar=self.path_to_image_tar,
                                             image_size=self.image_size,
                                             limit_dataset_size_to=self.limit_dataset_size_to)

    def train_dataloader(self):
        return DataLoader(self.data_,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )

    def val_dataloader(self):
        return DataLoader(self.data_,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )

    def test_dataloader(self):
        return DataLoader(self.data_,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers
                          )
