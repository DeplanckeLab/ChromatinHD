import pandas as pd
import chromatinhd as chd
import numpy as np


class TestSparse:
    def test_simple(self):
        path = "./hello"
        coords_pointed = {"a": pd.Index(["a", "b", "c"]), "b": pd.Index(["A", "B"])}
        coords_fixed = {"c": pd.Index(["1", "2", "3"]), "d": pd.Index(["100", "200"])}
        variables = {"x": ("a", "b", "c"), "y": ("a", "c", "d")}

        dataset = chd.sparse.SparseDataset2.create(path, variables, coords_pointed, coords_fixed)

        dataset["x"]["a", "B"] = np.array([1, 2, 3])
        dataset["x"]["c", "B"] = np.array([1, 0, 4])
        dataset["y"]["a"] = np.array([[1, 0], [2, 3], [1, 2]])

        dataset["x"]._read_data()[:]

        assert (dataset["x"]["a", "B"] == np.array([1.0, 2.0, 3.0])).all()
        assert (dataset["x"]["a", "A"] == np.array([0.0, 0.0, 0.0])).all()
        assert (dataset["x"]["b", "B"] == np.array([0.0, 0.0, 0.0])).all()
        assert (dataset["x"]["c", "B"] == np.array([1, 0, 4])).all()
        assert (dataset["y"]["a"] == np.array([[1, 0], [2, 3], [1, 2]])).all()
