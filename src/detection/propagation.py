"""
Clustering and Label Propagation Module
Implements feature clustering and label propagation methods
"""
import warnings
from typing import Dict, List, Tuple

import numpy
import pandas
import sklearn


# Import feature extraction capabilities


class LabelPropagator:
    """
    Clustering-based error detection using feature similarity and label propagation.

    This class implements a clustering approach to error detection:
    1. Extract features for each column
    2. Cluster similar data points based on features
    3. Select representative samples from each cluster
    4. Use labeled representatives to propagate labels to cluster members
    """

    def __init__(self, features: Dict[str, pandas.DataFrame], cluster=None, sample_size=20):
        """
        初始化LabelPropagator，支持直接传入聚类对象或使用默认MiniBatchKMeans。

        Args:
            features: 特征字典
            cluster: sklearn聚类对象（如MiniBatchKMeans），若为None则默认MiniBatchKMeans(n_clusters=20)
        """
        feature_lengths = [len(df) for df in features.values()]
        if len(set(feature_lengths)) != 1:
            raise ValueError("Feature ...")

        self.features = features
        self.length = feature_lengths[0]

        if cluster is None:
            cluster = sklearn.cluster.MiniBatchKMeans(n_clusters=sample_size)
        self.cluster = cluster

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.cluster_results = {}
        self.representatives = {}
        self.propagated_labels = {}

    def _cluster_feature(self, feature_dataframe: pandas.DataFrame, verbose: bool = False) -> Tuple[
        numpy.ndarray, List[int]]:
        """
        Cluster features using the provided cluster object.

        Args:
            feature_dataframe: Feature vectors to cluster
            verbose: Whether to print clustering information

        Returns:
            Tuple of (cluster_labels, representative_indices)
        """
        if feature_dataframe.empty:
            raise ValueError("Empty feature dataframe provided for clustering.")
        if verbose:
            print("Clustering object:", self.cluster)
        cluster_labels = self.cluster.fit_predict(feature_dataframe.values)
        centroids = self.cluster.cluster_centers_

        cluster_num = numpy.max(cluster_labels) + 1
        if verbose:
            print(f"Number of clusters formed: {cluster_num}")

        # Find representative samples for each cluster
        representatives = []
        for cluster_id in range(cluster_num):
            cluster_indices = numpy.where(cluster_labels == cluster_id)[0]

            if len(cluster_indices) == 0:
                representatives.append(-1)
                if verbose:
                    print(f"Warning: Cluster {cluster_id} has no members.")
                continue
            elif len(cluster_indices) == 1:
                representatives.append(cluster_indices[0])
                continue

            # Find the point closest to centroid as representative
            cluster_points = feature_dataframe.iloc[cluster_indices]
            distances = numpy.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
            closest_idx_in_cluster = numpy.argmin(distances)
            representatives.append(cluster_indices[closest_idx_in_cluster])

        return cluster_labels, representatives

    def _cluster(self, verbose: bool = False):
        """
        Cluster the entire dataset and generate features for each column.

        Args:
            verbose: Whether to print progress information

        Returns:
            Tuple of (cluster_assignments_df, representatives_df)
        """
        for idx, column in enumerate(self.features.keys()):
            if verbose:
                print(f"\nProcessing column: {column}, {idx + 1} / {len(self.features.keys())}")

            # Perform clustering
            clusters, representatives = self._cluster_feature(
                self.features[column],
                verbose=verbose
            )

            self.representatives[column] = representatives
            self.cluster_results[column] = clusters

            if verbose:
                print(
                    f"Clustered column '{column}': {len(representatives)} representatives, {numpy.max(clusters) + 1} clusters")

    def sample(self):
        if not self.cluster_results or not self.representatives:
            self._cluster()
        cluster_dataframe = pandas.DataFrame(self.cluster_results)
        representatives_dataframe = pandas.DataFrame(self.representatives)

        return cluster_dataframe, representatives_dataframe

    def propagate_column_labels(self, column_name: str, representative_labels: Dict[int, bool]):
        """
        Propagate labels based on clustering results.

        Args:
            representative_labels: Error labels for representatives

        Returns:
            Propagated labels for all data points
        """

        self.propagated_labels[column_name] = numpy.zeros(self.length, dtype=bool)

        for cluster_id, rep_idx in enumerate(self.representatives[column_name]):
            if rep_idx == -1:
                continue

            label = representative_labels[rep_idx]
            if rep_idx not in representative_labels:
                warnings.warn(f"Sample {rep_idx} is not labeled, set to False.")
                label = False
            # Propagate label to all members of this cluster
            cluster_indices = numpy.where(self.cluster_results[column_name] == cluster_id)[0]
            self.propagated_labels[column_name][cluster_indices] = label

    def propagate_dataset_labels(self, label_dataframe: pandas.DataFrame):
        if set(label_dataframe.columns) != set(self.features.keys()):
            raise ValueError("column size doesn't match")
        if len(label_dataframe) != self.length:
            raise ValueError("length doesn't match")

        cluster_dataframe, representatives_dataframe = self.sample()
        for column in label_dataframe.columns:
            representatives = representatives_dataframe[column].values
            labels = label_dataframe[column]

            selected_labels = {int(rep): bool(labels.iloc[rep])
                               for rep in representatives
                               if rep != -1 and 0 <= rep < self.length}
            self.propagate_column_labels(column, selected_labels)

        return self.get_prediction()

    def get_prediction(self):
        if set(self.propagated_labels.keys()) != set(self.features.keys()):
            warnings.warn("Not full prediction!")
        prediction_dataframe = pandas.DataFrame(self.propagated_labels)
        return prediction_dataframe
