from __future__ import annotations

import numpy as np
import pandas as pd
import pathlib
from typing import Union

from chromatinhd.flow import Flow, Stored, StoredDict, TSV
from chromatinhd.flow.tensorstore import Tensorstore
from chromatinhd import sparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scanpy as sc


class TimeTranscriptome(Flow):
    """
    A transcriptome during pseudotime
    """

    var: pd.DataFrame = TSV(index_name="gene")
    obs: pd.DataFrame = TSV(index_name="pseudocell")

    @classmethod
    def from_transcriptome(
        cls,
        gradient,
        transcriptome,
        path: Union[pathlib.Path, str] = None,
        overwrite=False,
    ):
        """
        Create a TimeTranscriptome object from a Transcriptome object.
        """

        raise NotImplementedError()

    layers = StoredDict(Tensorstore, kwargs=dict(dtype="<f4"))
    "Dictionary of layers, such as raw, normalized and imputed data."
