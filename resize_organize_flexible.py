import glob
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps
import traceback
from numpy import asarray
from albumentations import CenterCrop
from tqdm.contrib.concurrent import process_map
import numpy as np
import shutil

# Load metadata etc
filenames_and_labels = r"C:\Users\Rodney\PycharmProjects\Thesis_cur-AI-tor\notebooks\portrait_cc_14_r23n23.lisa.surfsara.nl_48373_(960, 40, 'euclidean', 1, 100).csv"
target_column_name = 'class'
image_input_folder = r'C:\Users\Rodney\Desktop\saatchi\saatchi'
# image_input_folder = r'E:\temp\thesisdata\micro_dataset1'
image_output_folder = r'C:\Users\Rodney\Desktop\saatchi\umap_hdsbscan_test_portrait_(960, 40, euclidean, 1, 100)_res'
resize_and_crop_ = False
size_ = 128
image_count_per_class = 1000000

cropper = CenterCrop(height=size_, width=size_)

# Load target data
targets_df = pd.read_csv(filenames_and_labels, index_col=0)

# Remove unnecessary columns
for col in targets_df.columns:
    if col != target_column_name:
        targets_df.drop(col, axis=1, inplace=True)

# Remove duplicates
targets_df = pd.DataFrame(targets_df.reset_index().
                          drop_duplicates(subset=['index'])).\
                          set_index('index')


def resize_pad_crop_image(input_path: str,
                          output_path: str,
                          desired_size: int,
                          mode: str,
                          ):
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
            img = asarray(img)
            img = cropper(image=img)
            full_output_path = output_path_ / filename
            img = Image.fromarray(img['image'])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(full_output_path)
        except (OSError, IOError):
            print(f'Opening image failed: \n {traceback.format_exc()}')
    elif mode == 'move':
        full_output_path = output_path_ / filename
        shutil.copy(input_path, full_output_path)


# Create list with unique class labels
label_folder_list = list(np.unique(targets_df[target_column_name].values))
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
            label = targets_df.loc[filename][target_column_name]
            if counter[label] < image_count_per_class:
                image_output_folder_with_label = image_output_folder + '\\' + str(label)
                resize_pad_crop_image(file, image_output_folder_with_label, size_, mode='move')
                counter.update({label: counter[label] + 1})
    except KeyError:
        # print(f'Label not found for file {file}, skipping!')
        pass
    except OSError:
        if filename is None:
            filename = file
        print(f'Skipping file {filename} due to OSError encountered: {traceback.format_exc()}, skipping!')


if __name__ == '__main__':
    filelist = glob.glob(image_input_folder + '*/*')
    r = process_map(run, filelist, max_workers=2, chunksize=10)
