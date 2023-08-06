import pandas as pd

from chromatinhd.flow import Flow, Stored, StoredDataFrame


class Clustering(Flow):
    labels = Stored()
    "Labels for each cell."

    cluster_info = StoredDataFrame(index_name="cluster")
    "Dataframe containing information for each cluster, such as a label."

    @classmethod
    def from_labels(cls, labels, path=None):
        clustering = cls(path)
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels).astype("category")
        elif not labels.dtype.name == "category":
            labels = labels.astype("category")
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
