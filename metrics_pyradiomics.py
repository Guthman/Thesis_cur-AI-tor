import numpy as np
from radiomics import featureextractor
import SimpleITK as sitk
from PIL import Image, ImageCms
import traceback

# Settings
enabled_feature_classes = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings.update({'voxelBatch ': 100})
extractor.disableAllFeatures()
for feature_class in enabled_feature_classes:
    extractor.enableFeatureClassByName(feature_class)


def load_image(input_path: str):
    img = None
    try:
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except (OSError, IOError):
        print(f'Opening image failed: \n {traceback.format_exc()}')
    return np.array(img)


def tile_image(img, n_blocks=(3, 3)):
    horizontal = np.array_split(img, n_blocks[0])
    split_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
    return np.asarray(split_img, dtype=np.ndarray).reshape(n_blocks)


def split_image_rgb(image):
    # Channel order: R, G, B
    return np.dsplit(image,image.shape[-1])


def split_image_lab(image):
    # Channel order: L, a, b
    image = Image.fromarray(image)
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p = ImageCms.createProfile("LAB")
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    image = ImageCms.applyTransform(image, rgb2lab)
    return [np.array(x) for x in image.split()]


def full_image_metrics(image: np.array):
    # Load image and dummy mask
    im = sitk.GetImageFromArray(image)
    ma = sitk.GetImageFromArray(np.ones(image.shape, dtype='uint8'))
    ma.CopyInformation(im)

    # Extract features into clean dict
    ic()
    features = extractor.execute(im, ma, label=1)
    ic()
    stats_keys = [x for x in list(features.keys()) if x[0][0] == 'o']  # Remove not needed stats
    stats_dict = {k[9:]: float(features[k]) for k in stats_keys}  # Remove prefix
    return stats_dict
