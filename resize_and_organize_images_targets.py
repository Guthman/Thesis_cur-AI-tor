import glob
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

# Load metadata etc
filenames_and_labels = r'C:\Users\R\PycharmProjects\Thesis_cur-AI-tor\notebooks\data\saatchi_targets.csv'
delimiter = ','
image_input_folder = r'F:\temp\thesisdata\saatchi_micro_resized512\culture'
image_output_folder = r'F:\temp\thesisdata\saatchI-target-labeled-test'
size_ = 128

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


def resize_and_pad_image(input_path: str,
                         output_path: str,
                         desired_size: int):
    input_path_ = Path(input_path)
    output_path_ = Path(output_path)

    assert input_path_.is_file()
    assert output_path_.is_dir(), print('Supplied output path is not a directory:' + output_path_.__str__())

    if input_path_.stat().st_size > 0:
        pass
    else:
        print(f'Filesize is 0, skipping file: {input_path_}')
        return

    filename = input_path_.name
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

label_folder_list = [0, 1, 2, 3, 4]

# Create the folders
for folder in label_folder_list:
    Path(image_output_folder + '/' + str(folder)).mkdir(parents=True, exist_ok=True)

print('Resizing and moving files...')
for file in tqdm(glob.glob(image_input_folder + '*/*')):
    try:
        filename = Path(file).name
        label = targets_df.loc[filename]['LIKES_VIEWS_RATIO_BIN_IDX']
        image_output_folder_with_label = image_output_folder + '\\' + str(label)
        resize_and_pad_image(file, image_output_folder_with_label, size_)
    except KeyError:
        print(f'Label not found for file {file}, skipping!')
