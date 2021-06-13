import metrics_pyradiomics
import metrics
import glob
import ujson as json
import os
import socket
import sys
from tqdm.contrib.concurrent import process_map

import warnings
warnings.filterwarnings('ignore')

from icecream import ic, install
install()


# For local testing
if socket.gethostname() == 'DESKTOP-8NT3G6P':
    directory = r"F:\temp\thesisdata\saatchi_micro_resized256\airplane\\"
    files = glob.glob(directory + '*.*')
    start_at_file_number = 0
    stop_at_file_number = 2
    worker_count = 2
    nochunks = 1
    ic.enable()
else:
    directory = sys.argv[1]
    files = glob.glob(directory + '*.*')
    start_at_file_number = int(sys.argv[2])
    if int(sys.argv[3]) >= len(files):
        stop_at_file_number = len(files)-1
    else:
        stop_at_file_number = int(sys.argv[3])
    worker_count = int(sys.argv[4])
    nochunks = 2
    ic.disable()

print(f'Arguments: ')
print(sys.argv)

files = glob.glob(directory + '*.*')

metrics_function_list_tiles = [metrics.contrast_rms, metrics.contrast_tenengrad, metrics.fractal_dimension,
                               metrics.sharpness, metrics.sharpness_laplacian, metrics.color_dominant,
                               metrics.colorfulness, metrics.hue_mean, metrics.pixel_intensity_mean,
                               metrics.saturation_mean, metrics.entropy_shannon, metrics.k_complexity_bw,
                               metrics.k_complexity_lab_l, metrics.k_complexity_lab_a, metrics.k_complexity_lab_b]

metrics_function_list_tiles_channels = [metrics.contrast_rms, metrics.contrast_tenengrad, metrics.fractal_dimension,
                                        metrics.sharpness, metrics.sharpness_laplacian, metrics.pixel_intensity_mean,
                                        metrics.entropy_shannon]

stats_list = []


def main(path):
    image = metrics_pyradiomics.load_image(path)

    # Pyradiomic stats
    stats = {'path': path}
    ic()
    r, g, b = metrics_pyradiomics.split_image_rgb(image)

    ic()
    r_measurement = metrics_pyradiomics.full_image_metrics(r)
    ic()
    g_measurement = metrics_pyradiomics.full_image_metrics(g)
    ic()
    b_measurement = metrics_pyradiomics.full_image_metrics(b)
    stats.update({'red': r_measurement})
    stats.update({'green': g_measurement})
    stats.update({'blue': b_measurement})

    l, a, b_ = metrics_pyradiomics.split_image_lab(image)
    l_measurement = metrics_pyradiomics.full_image_metrics(l)
    a_measurement = metrics_pyradiomics.full_image_metrics(a)
    b_measurement = metrics_pyradiomics.full_image_metrics(b_)
    stats.update({f'l': l_measurement})
    stats.update({f'a': a_measurement})
    stats.update({f'b': b_measurement})

    stats_dict = {'path': path}

    # Flatten the dict
    ic()
    for k in list(stats.keys())[1:]:
        for stat in stats[k]:
            stats_dict[k + '_' + stat] = stats[k][stat]

    # Oude stats
    ic()
    for metric_function in metrics_function_list_tiles:
        output = None
        try:
            output = metric_function(image)
        except Exception:
            output = -999
        finally:
            if not output:
                output = -999
            stats_dict.update({metric_function.__name__: output})

    ic()
    for metric_function in metrics_function_list_tiles_channels:
        try:
            output_red = metric_function(r.squeeze())
            output_green = metric_function(g.squeeze())
            output_blue = metric_function(b.squeeze())

            output_l = metric_function(l)
            output_a = metric_function(a)
            output_b = metric_function(b_)
        except Exception:
            output_red = -999
            output_green = -999
            output_blue = -999
            output_l = -999
            output_a = -999
            output_b = -999

        stats_dict.update({f'red_' + metric_function.__name__: output_red})
        stats_dict.update({f'green_' + metric_function.__name__: output_green})
        stats_dict.update({f'blue_' + metric_function.__name__: output_blue})
        stats_dict.update({f'l_' + metric_function.__name__: output_l})
        stats_dict.update({f'a_' + metric_function.__name__: output_a})
        stats_dict.update({f'b_' + metric_function.__name__: output_b})

    # stats_list.append(stats_dict)
    ic()
    with open(f'./output/bloedzweetentranen_{str(socket.gethostname())}_{os.getpid()}.json', 'a') as file:
        file.write(json.dumps(stats_dict) + ',\n')


if __name__ == '__main__':
    r = process_map(main, files[start_at_file_number:stop_at_file_number], max_workers=worker_count, chunksize=nochunks)
