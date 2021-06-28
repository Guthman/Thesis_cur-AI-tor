import pandas as pd
from pathlib import Path
import numpy as np
import umap
import hdbscan
import json
import pickle
from tqdm.contrib.concurrent import process_map
import itertools
import os
import sys
import socket

dataset = sys.argv[1]
list_start = int(sys.argv[2])
list_end = int(sys.argv[3])
worker_count = int(sys.argv[4])

if dataset == 'engres':
    # Load ResNet + engineered features
    df_x = pd.read_csv(r'/home/rguthman/metrics_scripts/umap/saatchi_micro_engineered_resnet.csv', index_col=0)
    X = df_x.values
elif dataset == 'res':
    # Load ResNet features
    data_dir = r'/home/rguthman/metrics_scripts/umap/micro_dataset1_resnet18_output_identity.json'
    with open(data_dir, 'r') as f:
        data_dict_list = json.load(f)

    data_dict = {}
    for element in data_dict_list:
        data_dict.update(element)

    df_x = pd.DataFrame.from_dict(data_dict, orient='index')
    X = df_x.values
elif dataset == 'eng':
    # Load engineered features
    df_x = pd.read_csv(r'/home/rguthman/metrics_scripts/umap/saatchi_micro_engineered_resnet.csv', index_col=0)
    for col in df_x.columns:
        if type(col) == int:
            df_x.drop(col, axis=1, inplace=True)
    X = df_x.values


def create_embedding(data: np.array,
                     n_neighbors: int,
                     n_components: int,
                     metric: str = 'euclidean',
                     full_dataset: bool = False):

    if full_dataset:
        filename_prefix = 'macro_all_'
    else:
        filename_prefix = 'micro_all_'

    embedding_filename = Path(f'/home/rguthman/metrics_scripts/umap/pkl_{dataset}/{str(socket.gethostname())}_{os.getpid()}_{filename_prefix}embedding_{n_neighbors}_{n_components}_{metric}.pkl')

    if embedding_filename.is_file():
        with open(embedding_filename, 'rb') as f:
            clusterable_embedding_ = pickle.load(f)

    else:
        clusterable_embedding_ = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=n_components,
            metric=metric,
            random_state=3,
        ).fit_transform(data)
        with open(embedding_filename, 'wb') as f:
            pickle.dump(clusterable_embedding_, f)
    return clusterable_embedding_


def get_clusters(clusterable_embedding_, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples,
                                min_cluster_size=min_cluster_size,
                                prediction_data=True).fit(clusterable_embedding_)
    soft_clusters_ = hdbscan.all_points_membership_vectors(clusterer)
    return soft_clusters_


n_neighbors_l = [30, 60, 120, 240, 480, 960, 1440]
n_components_l = [10, 20, 40, 80]
metrics_l = ['euclidean',
             'manhattan',
             'chebyshev',
             'minkowski',
             'canberra',
             'braycurtis',
             'mahalanobis',
             'wminkowski',
             'seuclidean',
             'cosine',
             'correlation',
             'haversine',
             'hamming',
             'jaccard',
             'dice',
             'russelrao',
             'kulsinski']
min_samples_l = [1, 5, 10, 20]
min_cluster_size_l = [100, 250, 500, 750, 1000, 1500]

ls = [n_neighbors_l, n_components_l, metrics_l, min_samples_l, min_cluster_size_l]
hparams_list = list(itertools.product(*ls))


def main(hparams):
    clusterable_embedding = create_embedding(data=X, n_neighbors=hparams[0], n_components=hparams[1], metric=hparams[2])
    soft_clusters = get_clusters(clusterable_embedding, min_samples=hparams[3], min_cluster_size=hparams[4])
    class_labels = [preds.argmax() for preds in soft_clusters]
    df_x_ = df_x.copy()
    df_x_['class'] = class_labels
    df_x_ = df_x_[['class']]

    # classes = df_x_.value_counts().__dict__
    class_count = str(len(df_x_.value_counts()))
    df_x_.to_csv(f'/home/rguthman/metrics_scripts/umap/output_{dataset}/cc_{class_count}_{str(socket.gethostname())}_{os.getpid()}_{str(hparams)}.csv')
    with open(f'/home/rguthman/metrics_scripts/umap/stats_{dataset}/{str(socket.gethostname())}_{os.getpid()}_embedding_stats.txt', 'a') as f:
        f.write(f'{str(hparams)}, class count: {class_count}\n')


if __name__ == '__main__':
    r = process_map(main, hparams_list[list_start:list_end], max_workers=worker_count, chunksize=1)
