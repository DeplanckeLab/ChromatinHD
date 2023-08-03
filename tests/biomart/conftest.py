import chromatinhd as chd
import pytest

import pathlib


@pytest.fixture(scope="module")
def dataset_grch38():
    return chd.biomart.Dataset.from_genome("GRCh38")
