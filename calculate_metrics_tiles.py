import metrics_pyradiomics
import metrics
import glob
import warnings
import ujson as json
import os
import socket
import sys
from tqdm.contrib.concurrent import process_map
warnings.filterwarnings('ignore')

directory = sys.argv[1]
start_at_file_number = int(sys.argv[2])
stop_at_file_number = int(sys.argv[3])

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
    tiled_image = metrics_pyradiomics.tile_image(image)

    tiles = []
    for i in range(len(tiled_image)):
        for j in tiled_image[i]:
            tiles.append(j)

    # Pyradiomic stats
    stats = {'path': path}
    for idx, tile in enumerate(tiles):
        r, g, b = metrics_pyradiomics.split_image_rgb(tile)

        r_measurement = metrics_pyradiomics.full_image_metrics(r)
        g_measurement = metrics_pyradiomics.full_image_metrics(g)
        b_measurement = metrics_pyradiomics.full_image_metrics(b)
        stats.update({f'tile{idx + 1}_red': r_measurement})
        stats.update({f'tile{idx + 1}_green': g_measurement})
        stats.update({f'tile{idx + 1}_blue': b_measurement})

        # l, a, b = metrics_pyradiomics.split_image_lab(tile)
        # l_measurement = metrics_pyradiomics.full_image_metrics(l)
        # a_measurement = metrics_pyradiomics.full_image_metrics(a)
        # b_measurement = metrics_pyradiomics.full_image_metrics(b)
        # stats.update({f'tile{idx + 1}_l': l_measurement})
        # stats.update({f'tile{idx + 1}_a': a_measurement})
        # stats.update({f'tile{idx + 1}_b': b_measurement})

    stats_dict = {'path': path}

    for k in list(stats.keys())[1:]:
        for stat in stats[k]:
            stats_dict[k + '_' + stat] = stats[k][stat]

    # Oude stats
    for idx, tile in enumerate(tiles):
        for metric_function in metrics_function_list_tiles:
            output = None
            try:
                output = metric_function(tile)
            except Exception:
                output = -999
            finally:
                if not output:
                    output = -999
                stats_dict.update({f'tile{idx + 1}_' + metric_function.__name__: output})

        r, g, b = metrics_pyradiomics.split_image_rgb(tile)
        # l, a, b_ = metrics_pyradiomics.split_image_lab(tile)

        for metric_function in metrics_function_list_tiles_channels:

            try:
                output_red = metric_function(r.squeeze())
                output_green = metric_function(g.squeeze())
                output_blue = metric_function(b.squeeze())

                # output_l = metric_function(l)
                # output_a = metric_function(a)
                # output_b = metric_function(b_)
            except Exception:
                output_red = -999
                output_green = -999
                output_blue = -999
                # output_l = -999
                # output_a = -999
                # output_b = -999

            stats_dict.update({f'tile{idx + 1}_red_' + metric_function.__name__: output_red})
            stats_dict.update({f'tile{idx + 1}_green_' + metric_function.__name__: output_green})
            stats_dict.update({f'tile{idx + 1}_blue_' + metric_function.__name__: output_blue})
            # stats_dict.update({f'tile{idx + 1}_l_' + metric_function.__name__: output_l})
            # stats_dict.update({f'tile{idx + 1}_a_' + metric_function.__name__: output_a})
            # stats_dict.update({f'tile{idx + 1}_b_' + metric_function.__name__: output_b})

    # stats_list.append(stats_dict)
    with open(f'./output/bloedzweetentranen_{str(socket.gethostname())}_{os.getpid()}.json', 'a') as file:
        file.write(json.dumps(stats_dict) + ',\n')


if __name__ == '__main__':
    r = process_map(main, files[start_at_file_number:stop_at_file_number], max_workers=48, chunksize=5)
