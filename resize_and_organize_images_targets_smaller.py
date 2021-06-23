import glob
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import traceback
from numpy import asarray
from albumentations import CenterCrop
from tqdm.contrib.concurrent import process_map
# import numpy as np

# Load metadata etc
filenames_and_labels = r'C:\Users\Rodney\PycharmProjects\Thesis_cur-AI-tor\notebooks\data\saatchi_targets.csv'
image_input_folder = r'C:\Users\Rodney\Desktop\saatchi\portrait512\portrait512'
image_output_folder = r'E:\temp\thesisdata\saatchi_portrait_cond128'
size_ = 128
image_count_per_class = 1000000

cropper = CenterCrop(height=size_, width=size_)

# Load target data
targets_df = pd.read_csv(filenames_and_labels, header=None)
targets_df.columns = ['FILENAME', 'PRICE', 'LIKES_VIEWS_RATIO']
# Bin the values
targets_df['PRICE_BIN_IDX'] = pd.qcut(targets_df['PRICE'], q=5, labels=[0, 1, 2, 3, 4])
targets_df['LIKES_VIEWS_RATIO_BIN_IDX'] = pd.qcut(targets_df['LIKES_VIEWS_RATIO'], q=5, labels=[0, 1, 2, 3, 4])
targets_df = targets_df.astype({'PRICE_BIN_IDX': int, 'LIKES_VIEWS_RATIO_BIN_IDX': int})
targets_df.drop(['PRICE', 'LIKES_VIEWS_RATIO'], axis=1, inplace=True)
targets_df.set_index('FILENAME', inplace=True)
targets_df.drop('PRICE_BIN_IDX', axis=1, inplace=True)
targets_df = pd.DataFrame(targets_df.reset_index().drop_duplicates(subset=['FILENAME'])).set_index('FILENAME')


def resize_pad_crop_image(input_path: str,
                          output_path: str,
                          desired_size: int,
                          mode: str):
    input_path_ = Path(input_path)
    output_path_ = Path(output_path)

    assert input_path_.is_file()
    assert output_path_.is_dir(), print('Supplied output path is not a directory:' + output_path_.__str__())

    if input_path_.stat().st_size > 0:
        pass
    else:
        print(f'Filesize is 0, skipping file: {input_path_}')
        return

    filename = input_path_.name.replace('\n', '')

    if mode == 'pad':
        try:
            img = Image.open(input_path)
            old_size = img.size
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img = img.resize(new_size, Image.ANTIALIAS)

            # create a new image and paste the resized on it
            new_img = Image.new('RGB', (desired_size, desired_size))
            new_img.paste(img, ((desired_size - new_size[0]) // 2,
                                (desired_size - new_size[1]) // 2))

            full_output_path = output_path_ / filename
            new_img.save(full_output_path)
        except (OSError, IOError):
            print(f'Opening image failed: \n {traceback.format_exc()}')
    elif mode == 'crop':
        try:
            img = Image.open(input_path)
            if asarray(img).shape[0] < size_:
                img = ImageOps.pad(img, (size_, size_))
            elif asarray(img).shape[0] > size_:
                img = ImageOps.fit(img, (size_, size_))
            img = asarray(img)
            img = cropper(image=img)
            full_output_path = output_path_ / filename
            img = Image.fromarray(img['image'])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(full_output_path)
        except (OSError, IOError):
            print(f'Opening image failed: \n {traceback.format_exc()}')


label_folder_list = [0, 1, 2, 3, 4]
counter = {k: 0 for k in label_folder_list}

# Create the folders
for folder in label_folder_list:
    Path(image_output_folder + '/' + str(folder)).mkdir(parents=True, exist_ok=True)
print('Resizing and moving files...')


def run(file):
    filename = None
    try:
        if all(count >= image_count_per_class for count in counter.values()):
            return
        else:
            filename = Path(file).name
            label = targets_df.loc[filename]['LIKES_VIEWS_RATIO_BIN_IDX']
            if counter[label] < image_count_per_class:
                image_output_folder_with_label = image_output_folder + '\\' + str(label)
                resize_pad_crop_image(file, image_output_folder_with_label, size_, mode='crop')
                counter.update({label: counter[label] + 1})
    except KeyError:
        print(f'Label not found for file {file}, skipping!')
    except OSError:
        if filename is None:
            filename = file
        print(f'Skipping file {filename} due to OSError encountered: {traceback.format_exc()}, skipping!')


if __name__ == '__main__':
    filelist = glob.glob(image_input_folder + '*/*')
    r = process_map(run, filelist, max_workers=9, chunksize=10)
