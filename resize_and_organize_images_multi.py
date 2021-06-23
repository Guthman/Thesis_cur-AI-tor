import glob
import csv
from numpy import asarray
from pathlib import Path
from PIL import Image, ImageOps
from albumentations import CenterCrop
from tqdm.auto import tqdm
import traceback

# Load metadata etc
# filenames_and_labels = r'F:\temp\thesisdata\SAATCHI_DATASET_FULL.tsv'
filenames_and_labels = r'C:\Users\Rodney\PycharmProjects\stylegan2-ada\rmg_utils\SAATCHI_DATASET_FULL.tsv'
delimiter = '\t'
# image_input_folder = r'C:\Users\R\PycharmProjects\Thesis_Saatchi_scraper\micro_dataset1'
image_input_folder = r'C:\Users\Rodney\Desktop\saatchi\saatchi'
# image_output_folder = r'E:\temp\thesisdata\saatchi'
image_output_folder = r'C:\Users\Rodney\Desktop\saatchi\peeps128'
size_ = 128
max_image_count = 9999999
# label_folder_list_override = ['animal', 'cows', 'horse', 'cats', 'dogs', 'fish']
label_folder_list_override = ['portrait', 'people', 'men', 'women',
                              'celebrity', 'children', 'fashion',
                              'popular_culture', 'pop_culture_celebrity'
                              ]

cropper = CenterCrop(height=size_, width=size_)


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

    filename = input_path_.name

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


# Create a dict with all filenames and associated labels
with open(filenames_and_labels, 'rt')as f:
    data = list(csv.reader(f, delimiter=delimiter))
    file_dict = {}
    for row in data:
        file_dict.update({row[0]: row[1]})

# Create list of sanitized labels to be used as folder names
label_folder_list = [s.replace(' ', '_')
                     .replace('&', '_')
                     .replace('/', '_')
                     .replace('__', '_')
                     .replace('__', '_')
                     .lower()
                     for s in set(file_dict.values())]

# Create dict for lookup up the correct folder name given a label
label_folder_lookup = {}

for s in set(file_dict.values()):
    label_folder_lookup.update({s: s.replace(' ', '_')
                               .replace('&', '_')
                               .replace('/', '_')
                               .replace('__', '_')
                               .replace('__', '_')
                               .lower()})
print(f'Lookup dict: {label_folder_lookup}')

label_folder_list = label_folder_list_override

# Create counter dict
labeled_image_counter = {x: 0 for x in label_folder_list}


def main(label_folder_list_):
    # Create the folders
    for folder in label_folder_list_:
        Path(image_output_folder + '/' + folder).mkdir(parents=True, exist_ok=True)

    print('Resizing and moving files...')
    for file in tqdm(glob.glob(image_input_folder + '*/*')):
        try:
            label = label_folder_lookup[file_dict[Path(file).name]]
            image_output_folder_with_label = image_output_folder + '\\' + label
            if label in label_folder_list_:
                resize_pad_crop_image(file, image_output_folder_with_label, size_, mode='crop')
                labeled_image_counter.update({label: labeled_image_counter[label]+1})
            if labeled_image_counter.get(label) == max_image_count:
                break
        except KeyError:
            print(f'Label not found for file {file}, skipping! \n {traceback.format_exc()}')


main(label_folder_list)
