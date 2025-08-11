import time

from ed.init import Dataset
from evaluate.evaluation import evaluate_prediction
from features import *

start_time = time.time()
# random.seed(42)

# dataset_name = 'flights'
# dataset_name = 'movies_1'
# dataset_name = 'billionaire'
# dataset_name = 'beers'
# dataset_name = 'hospital'
# dataset_name = 'rayyan'
dataset_name = 'spotify'

print('loading dataset:', dataset_name)
dataset = Dataset(dataset_name)
data_error = dataset.dirty_data
err_labels = dataset.error_labels

print(f"\n------------------- Data loaded in {time.time() - start_time:.2f} seconds -------------------")
start_time = time.time()

label_nums = [0, 1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
label_nums = [0, 20]
f1s = [0.0]
for label_num in label_nums[1:]:
    cluster_params = {
        'n_clusters': label_num,
        'random_state': 20,
    }
    f1s_tmp = []
    for _ in range(1):
        cluster_df, repre_df = cluster_dataset(data_error, cluster_params=cluster_params, verbose=True)
        pred_df = propagate_labels_from_clusters(cluster_df, repre_df, err_labels)
        evaluation, _ = evaluate_prediction(err_labels, pred_df)
        f1 = evaluation['overall']['f1']
        f1s_tmp.append(f1)
    avg_f1 = sum(f1s_tmp) / len(f1s_tmp)
    f1s.append(avg_f1)
    print(f"Avg F1 score for label number {label_num}: {avg_f1:.5f}")

print(f"\nF1 scores: {f1s}")

# cluster_params = {
#     'n_clusters': 20,
# }
# cluster_df, repre_df = cluster_dataset(data_error, cluster_params=cluster_params, verbose=False)
#
# print(f"\n--------------- Clustering completed in {time.time() - start_time:.2f} seconds ---------------")
# start_time = time.time()
#
# pred_df = propagate_labels_from_clusters(cluster_df, repre_df, err_labels)
# print(f"\n------------ Label propagation completed in {time.time() - start_time:.2f} seconds ------------")
#
# evaluate_model(err_labels, pred_df)
