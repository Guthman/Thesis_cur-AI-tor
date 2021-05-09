import glob
# import os
from cv2 import imread
import gzip
import metrics
import ujson
# from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
# from tqdm.contrib.concurrent import thread_map
from threading import get_ident

files = glob.glob('E:/temp/thesisdata/micro_dataset1/*.*')


def calculate_metrics(image):
    img = imread(image)
    with gzip.open(f'./calculated_metrics/haar_wavelet_{get_ident()}.json.gz', 'a') as file:
        d = {image.replace('E:/temp/thesisdata/micro_dataset1\\', ''):
            {
                'haar_wavelet_h': metrics.haar_wavelet(img)[1][0].astype('uint8').tolist(),
                'haar_wavelet_v': metrics.haar_wavelet(img)[1][1].astype('uint8').tolist(),
                'haar_wavelet_d': metrics.haar_wavelet(img)[1][2].astype('uint8').tolist()
            }
        }

        file.write(str(ujson.dumps(d) + ',\n').encode('utf-8'))


if __name__ == '__main__':
    # r = thread_map(calculate_metrics, files, max_workers=12, chunksize=5)
    r = process_map(calculate_metrics, files, max_workers=6, chunksize=1)
