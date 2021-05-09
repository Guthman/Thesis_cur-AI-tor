from cv2 import imread
import metrics_numba as metrics
import ujson
from threading import get_ident
import warnings
import pickle
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

with open('files.pkl', 'rb') as f:
    files = pickle.load(f)

for image in tqdm(files[:36]):
    img = imread(image)
    with open(f'./calculated_metrics/macro_dataset1_all_except_brisque_shannon_kcomplexity_{get_ident()}.json', 'a') as file:
        file.write(
            ujson.dumps(
                {image.replace('C:/Users/Rodney/Desktop/saatchi/saatchi\\', ''):
                    {
                        'contrast_rms': str(round(metrics.contrast_rms(img), 4)),
                        'contrast_tenengrad': str(round(metrics.contrast_tenengrad(img), 6)),
                        'fractal_dimension': str(round(metrics.fractal_dimension(img), 6)),
                        'sharpness': str(round(metrics.sharpness(img), 4)),
                        'sharpness_laplacian': str(round(metrics.sharpness_laplacian(img), 4)),
                        # 'brisque_score': str(round(metrics.brisque_score(img), 4)),
                        'color_dominant': metrics.color_dominant(img),
                        'colorfulness': str(round(metrics.colorfulness(img), 4)),
                        'pixel_intensity_mean': str(round(metrics.pixel_intensity_mean(img), 4)),
                        'saturation_mean': str(round(metrics.saturation_mean(img), 4)),
                        # 'entropy_shannon': str(round(metrics.entropy_shannon(img), 4)),
                        # 'k_complexity_bw': str(round(metrics.k_complexity_bw(img), 4)),
                        # 'k_complexity_lab_l': str(round(metrics.k_complexity_lab_l(img), 4)),
                        # 'k_complexity_lab_a': str(round(metrics.k_complexity_lab_a(img), 4)),
                        # 'k_complexity_lab_b': str(round(metrics.k_complexity_lab_b(img), 4))
                    }
                }
            ) + ',\n')
