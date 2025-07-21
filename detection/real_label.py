import random
import time

from dataset import Dataset
from evaluation.evaluation import evaluate_model
from features import *

start_time = time.time()
random.seed(42)

# dataset_name = 'flights'
# dataset_name = 'movies'
# dataset_name = 'movies_1'
# dataset_name = 'billionaire'
# dataset_name = 'beers'
dataset_name = 'hospital'
# dataset_name = 'rayyan'
# dataset_name = 'tax50k'
# dataset_name = 'tax'

print('loading dataset:', dataset_name)
# data_clean = pd.read_csv(f'../data/{dataset_name}_clean.csv', dtype=str)
# data_error = pd.read_csv(f'../data/{dataset_name}_error-01.csv', dtype=str)
dataset = Dataset(dataset_name)
data_error = dataset.dirty_data
err_labels = dataset.error_labels

print(f"\n------------------- Data loaded in {time.time() - start_time:.2f} seconds -------------------")
start_time = time.time()

cluster_params = {
    'n_clusters': 20,
}
cluster_df, repre_df = cluster_dataset(data_error, cluster_params=cluster_params, verbose=True)
# print(cluster_df, repre_df)
print(f"\n--------------- Clustering completed in {time.time() - start_time:.2f} seconds ---------------")
start_time = time.time()

pred_df = propagate_labels_from_clusters(cluster_df, repre_df, err_labels)
print(f"\n------------ Label propagation completed in {time.time() - start_time:.2f} seconds ------------")

evaluate_model(err_labels, pred_df)
