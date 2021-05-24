import glob
from cv2 import imread
import metrics
import ujson
# from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
# from tqdm.contrib.concurrent import thread_map
from threading import get_ident
import pickle
import warnings

warnings.filterwarnings('ignore')

files = glob.glob('C:/Users/R/PycharmProjects/Thesis_cur-AI-tor/images/*.*')
with open('files.pkl', 'wb') as f:
    pickle.dump(files, f)
# with open('files.pkl', 'rb') as f:
#     files = pickle.load(f)[:4]

# def calculate_metrics(image):
#     img = imread(image)
#     with gzip.open(f'./calculated_metrics/haar_wavelet_{get_ident()}.json.gz', 'a') as file:
#         d = {image.replace('E:/temp/thesisdata/micro_dataset1\\', ''):
#             {
#                 'haar_wavelet_h': metrics.haar_wavelet(img)[1][0].astype('uint8').tolist(),
#                 'haar_wavelet_v': metrics.haar_wavelet(img)[1][1].astype('uint8').tolist(),
#                 'haar_wavelet_d': metrics.haar_wavelet(img)[1][2].astype('uint8').tolist()
#             }
#         }
#
#         file.write(str(ujson.dumps(d) + ',\n').encode('utf-8'))


def calculate_metrics(image):
    img = imread(image)
    metrics_values = {'filename': image}
    output = None
    metrics_function_list = [metrics.contrast_rms, metrics.contrast_tenengrad, metrics.fractal_dimension,
                             metrics.sharpness, metrics.sharpness_laplacian, metrics.brisque_score,
                             metrics.color_dominant, metrics.colorfulness, metrics.pixel_intensity_mean,
                             metrics.saturation_mean, metrics.entropy_shannon, metrics.k_complexity_bw,
                             metrics.k_complexity_lab_l, metrics.k_complexity_lab_a, metrics.k_complexity_lab_b]

    for metric_function in metrics_function_list:
        try:
            output = metric_function(img)
        except Exception:
            output = -999
        finally:
            if not output:
                output = -999
            metrics_values.update({metric_function.__name__: output})

    # with open(f'./calculated_metrics/macro_dataset1_all_except_brisque_shannon_kcomplexity_{get_ident()}.json', 'a') as file:
    with open(f'./calculated_metrics/test_all_{get_ident()}.json', 'a') as file:
        file.write(ujson.dumps(metrics_values) + ',\n')

if __name__ == '__main__':
    # r = thread_map(calculate_metrics, files, max_workers=12, chunksize=5)
    r = process_map(calculate_metrics, files, max_workers=2, chunksize=1)
