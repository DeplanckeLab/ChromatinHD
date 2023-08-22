from __future__ import annotations
import pandas as pd
import numpy as np

from chromatinhd.flow import Flow, Stored, StoredDataFrame, PathLike
from chromatinhd.flow.tensorstore import Tensorstore


class Clustering(Flow):
    labels: pd.DataFrame = Stored()
    "Labels for each cell."

    indices: np.array = Tensorstore(dtype=">i4")
    "Indices for each cell."

    var: pd.DataFrame = StoredDataFrame(index_name="cluster")
    "Information for each cluster, such as a label, color, ..."

    @classmethod
    def from_labels(cls, labels: pd.Series, path: PathLike = None) -> Clustering:
        """
        Create a Clustering object from a series of labels.

        Parameters:
            labels:
                Series of labels for each cell, with index corresponding to cell
                names.
            path:
                Folder where the clustering information will be stored.

        Returns:
            Clustering object.

        """
        clustering = cls(path)
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels).astype("category")
        elif not labels.dtype.name == "category":
            labels = labels.astype("category")
        clustering.labels = labels
        clustering.indices = labels.cat.codes.values
        clustering.var = (
            pd.DataFrame(
                {
                    "cluster": labels.unique(),
                    "n_cells": labels.value_counts(),
                    "label": labels.unique(),
                }
            )
            .set_index("cluster")
            .loc[labels.cat.categories]
        )
        return clustering

    @property
    def n_clusters(self):
        return len(self.labels.unique())

    # temporarily link cluster_info to var
    @property
    def cluster_info(self):
        return self.var

    @cluster_info.setter
    def cluster_info(self, cluster_info):
        self.var = cluster_info
