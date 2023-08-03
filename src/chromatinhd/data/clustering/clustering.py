import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import Flow, Stored, StoredDict, StoredDataFrame
from chromatinhd import sparse
from chromatinhd.utils import Unpickler


class Clustering(Flow):
    labels = Stored("labels")
    "Labels for each cell."

    cluster_info = StoredDataFrame("cluster_info")
    "Dataframe containing information for each cluster, such as a label."

    @classmethod
    def from_labels(cls, labels, path):
        clustering = cls(path)
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels).astype("category")
        clustering.labels = labels
        clustering.cluster_info = (
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
