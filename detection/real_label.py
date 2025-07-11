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

# cluster_params = {
#     't': 100,
#     'criterion': 'maxclust',
# }

# cluster_params = {
#     't': 0.5,
#     'criterion': 'distance',
# }

cluster_params = None

pred_df = cluster_and_propagate(data_error, err_labels, cluster_params=cluster_params, verbose=True)
# null_cnt = pred_df.isnull().sum().sum()
# print(null_cnt)
evaluate_model(err_labels, pred_df)
