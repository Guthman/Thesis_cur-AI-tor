import glob
# import os
from cv2 import imread
import metrics
import ujson
# from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.concurrent import thread_map
from threading import get_ident
import warnings

warnings.filterwarnings('ignore')

files = glob.glob('E:/temp/thesisdata/micro_dataset1/*.*')


def calculate_metrics(image):
    img = imread(image)
    with open(f'./calculated_metrics/k_complexity_bw_{get_ident()}.json', 'a') as file:
        file.write(
            ujson.dumps(
                {image.replace('E:/temp/thesisdata/micro_dataset1\\', ''):
                    {
                        # 'contrast_rms': str(round(metrics.contrast_rms(img), 4)),
                        # 'contrast_tenengrad': str(round(metrics.contrast_tenengrad(img), 6)),
                        # 'fractal_dimension': str(round(metrics.fractal_dimension(img), 6)),
                        # 'sharpness': str(round(metrics.sharpness(img), 4)),
                        # 'sharpness_laplacian': str(round(metrics.sharpness_laplacian(img), 4)),
                        # 'brisque_score': str(round(metrics.brisque_score(img), 4)),
                        # 'color_dominant': metrics.color_dominant(img),
                        # 'colorfulness': str(round(metrics.colorfulness(img), 4)),
                        # 'pixel_intensity_mean': str(round(metrics.pixel_intensity_mean(img), 4)),
                        # 'saturation_mean': str(round(metrics.saturation_mean(img), 4)),
                        # 'entropy_shannon': str(round(metrics.entropy_shannon(img), 4)),
                        'k_complexity_bw': str(round(metrics.k_complexity_bw(img), 4)),
                        # 'k_complexity_lab_l': str(round(metrics.k_complexity_lab_l(img), 4)),
                        # 'k_complexity_lab_a': str(round(metrics.k_complexity_lab_a(img), 4)),
                        # 'k_complexity_lab_b': str(round(metrics.k_complexity_lab_b(img), 4))
                    }
                 }
            ) + ',\n')


if __name__ == '__main__':
    # r = thread_map(calculate_metrics, files, max_workers=10, chunksize=1)
    r = process_map(calculate_metrics, files, max_workers=11, chunksize=1)
