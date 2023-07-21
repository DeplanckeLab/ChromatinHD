import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import Flow, Stored, StoredDict
from chromatinhd import sparse
from chromatinhd.utils import Unpickler


class Clustering(Flow):
    labels = Stored("labels")

    @classmethod
    def from_labels(cls, labels, path):
        clustering = cls(path)
        clustering.labels = labels
        return clustering
