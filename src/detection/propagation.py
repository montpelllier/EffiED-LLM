"""
Clustering and Label Propagation Module
Implements feature clustering and label propagation methods
"""
import warnings
from typing import Dict

import numpy as np
import sklearn
from pandas import DataFrame


class LabelPropagator:
    """
    Clustering-based error detection using feature similarity and label propagation.

    This class implements a clustering approach to error detection:
    1. Extract features for each column
    2. Cluster similar data points based on features
    3. Select representative samples from each cluster
    4. Use labeled representatives to propagate labels to cluster members
    """

    def __init__(self, features: Dict[str, DataFrame], cluster=None, sample_size=20, seed=20):
        """
        Initialize LabelPropagator with support for custom clustering object or default MiniBatchKMeans.

        Args:
            features: Dictionary of features for each column
            cluster: sklearn clustering object (e.g., MiniBatchKMeans), defaults to MiniBatchKMeans(n_clusters=20)
            sample_size: Number of clusters when using default clustering
            seed: Random seed for reproducible clustering
        """
        feature_lengths = [len(df) for df in features.values()]
        if len(set(feature_lengths)) != 1:
            raise ValueError("Feature ...")

        self.features = features
        self.length = feature_lengths[0]

        if cluster is None:
            cluster = sklearn.cluster.MiniBatchKMeans(n_clusters=sample_size, random_state=seed)
            print(f"Using clustering algorithm: {cluster}")
        self.cluster = cluster

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.cluster_results = {}
        self.representatives = {}
        self.propagated_labels = {}

    def _cluster_feature(self, feature_dataframe: DataFrame, verbose: bool = False):
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

        cluster_num = np.max(cluster_labels) + 1
        if verbose:
            print(f"Number of clusters formed: {cluster_num}")

        # Find representative samples for each cluster
        representatives = []
        for cluster_id in range(cluster_num):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]

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
            distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
            closest_idx_in_cluster = np.argmin(distances)
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
                    f"Clustered column '{column}': {len(representatives)} representatives, {np.max(clusters) + 1} clusters")

    def sample(self):
        """Sample the clustering centriods."""
        if not self.cluster_results or not self.representatives:
            self._cluster()
        cluster_dataframe = DataFrame(self.cluster_results)
        representatives_dataframe = DataFrame(self.representatives)

        return cluster_dataframe, representatives_dataframe

    def propagate_column_labels(self, column_name: str, representative_labels: Dict[int, bool]):
        """
        Propagate labels from representative samples to all cluster members for a specific column.

        Args:
            column_name: Name of the column to propagate labels for
            representative_labels: Dictionary mapping representative sample indices to their labels
        """

        self.propagated_labels[column_name] = np.zeros(self.length, dtype=bool)

        for cluster_id, rep_idx in enumerate(self.representatives[column_name]):
            if rep_idx == -1:
                continue

            label = representative_labels[rep_idx]
            if rep_idx not in representative_labels:
                warnings.warn(f"Sample {rep_idx} is not labeled, set to False.")
                label = False
            # Propagate label to all members of this cluster
            cluster_indices = np.where(self.cluster_results[column_name] == cluster_id)[0]
            self.propagated_labels[column_name][cluster_indices] = label

    def propagate_dataset_labels(self, label_dataframe: DataFrame):
        """
        Propagate labels for the entire dataset using clustering results.

        Args:
            label_dataframe: DataFrame containing true labels for representative samples

        Returns:
            DataFrame with propagated predictions for all samples

        Raises:
            ValueError: If column names don't match or dataset length doesn't match
        """
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
        """Get the propagation result."""
        if set(self.propagated_labels.keys()) != set(self.features.keys()):
            warnings.warn("Not full prediction!")
        prediction_dataframe = DataFrame(self.propagated_labels)
        return prediction_dataframe
