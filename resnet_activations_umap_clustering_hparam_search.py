from tqdm.contrib.concurrent import process_map

import umap
import hdbscan
import pandas as pd
import itertools
import numpy as np
import ujson as json

# load ResNet activations
data_dir = r'C:\Users\Rodney\PycharmProjects\Thesis_cur-AI-tor\notebooks\micro_dataset1_resnet18_output_identity.json'
with open(data_dir, 'r') as f:
    data_dict_list = json.load(f)

data_dict = {}
for element in data_dict_list:
    data_dict.update(element)

df_x = pd.DataFrame.from_dict(data_dict, orient='index')
X = df_x.values

# Define hparam list
n_neighbors_l = [60, 120, 240, 480, 960]
n_components_l = [10, 20, 40, 80, 160]
min_samples_l = [10, 20, 40, 80, 160]
min_cluster_size_l = [500, 750, 1000, 1250, 1500]

ls = [n_neighbors_l, n_components_l, min_samples_l, min_cluster_size_l]
hparams_list = list(itertools.product(*ls))
hparams_list.reverse()


def calculate(hparams):
    clusterable_embedding = umap.UMAP(
        n_neighbors=hparams[0],
        min_dist=0.0,
        n_components=hparams[1],
        random_state=3,
    ).fit_transform(X)

    labels = hdbscan.HDBSCAN(
        min_samples=hparams[2],
        min_cluster_size=hparams[3],
    ).fit_predict(clusterable_embedding)

    clustered = (labels >= 0)

    d = {hparams: {
        'embedding': clusterable_embedding,
        'metric': np.sum(clustered) / X.shape[0]}}
    print(f'hparams: {hparams}, metric: {np.sum(clustered) / X.shape[0]}')


if __name__ == '__main__':
    r = process_map(calculate, hparams_list, max_workers=4, chunksize=2)
