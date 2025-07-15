from evaluation.evaluation import evaluate_model
from features import *

# dataset_name = 'flights'
# dataset_name = 'movies'
dataset_name = 'billionaire'
# dataset_name = 'beers'
# dataset_name = 'hospital'
# dataset_name = 'rayyan'
# dataset_name = 'tax50k'

data_clean = pd.read_csv(f'../data/{dataset_name}_clean.csv', dtype=str)
data_error = pd.read_csv(f'../data/{dataset_name}_error-01.csv', dtype=str)

err_labels = data_clean != data_error

cluster_params = {
    'n_clusters': 30,
}

# pred_df = cluster_and_propagate(data_error, err_labels, cluster_params=cluster_params, verbose=True)
cluster_df, repre_df = cluster_dataset(data_error, cluster_params=cluster_params)
# print(cluster_df, repre_df)
pred_df = propagate_labels_from_clusters(cluster_df, repre_df, err_labels, verbose=True)

evaluate_model(err_labels, pred_df)