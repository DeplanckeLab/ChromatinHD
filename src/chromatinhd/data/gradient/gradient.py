from __future__ import annotations
import pandas as pd
import numpy as np

from chromatinhd.flow import Flow, Stored, StoredDataFrame, PathLike
from chromatinhd.flow.tensorstore import Tensorstore


class Gradient(Flow):
    values: np.array = Tensorstore(dtype="<f4")
    "Values for each cell."

    var: pd.DataFrame = StoredDataFrame(index_name="gradient")
    "Information on each gradient, such as a label, color, ..."

    @classmethod
    def from_values(cls, values: pd.Series, path: PathLike = None) -> Gradient:
        """
        Create a Gradient object from a series of values.

        Parameters:
            values:
                Series of values for each cell, with index corresponding to cell
                names.
            path:
                Folder where the gradient information will be stored.

        Returns:
            Gradient object.

        """
        gradient = cls(path)
        if isinstance(values, pd.Series):
            values = pd.DataFrame({"gradient": values}).astype(float)
        elif not isinstance(values, pd.DataFrame):
            values = pd.DataFrame(values).astype(float)
        gradient.values = values.values
        gradient.var = pd.DataFrame(
            {
                "gradient": values.columns,
                "label": values.columns,
            }
        ).set_index("gradient")
        return gradient

    @property
    def n_gradients(self):
        return len(self.values.shape[1])
