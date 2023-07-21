import chromatinhd as chd
import pytest

import pathlib


@pytest.fixture(scope="session")
def example_dataset_folder(tmp_path_factory):
    example_dataset_folder = tmp_path_factory.mktemp("example")

    import pkg_resources
    import shutil

    DATA_PATH = pathlib.Path(
        pkg_resources.resource_filename("chromatinhd", "data/examples/pbmc10ktiny/")
    )

    # copy all files from data path to dataset folder
    for file in DATA_PATH.iterdir():
        shutil.copy(file, example_dataset_folder / file.name)
    return example_dataset_folder


@pytest.fixture(scope="session")
def example_transcriptome(example_dataset_folder):
    import scanpy as sc

    adata = sc.read(example_dataset_folder / "transcriptome.h5ad")
    transcriptome = chd.data.Transcriptome.from_adata(
        adata, path=example_dataset_folder / "transcriptome"
    )
    return transcriptome


@pytest.fixture(scope="session")
def example_clustering(example_dataset_folder, example_transcriptome):
    clustering = chd.data.Clustering.from_labels(
        example_transcriptome.adata.obs["celltype"],
        path=example_dataset_folder / "clustering",
    )
    return clustering


@pytest.fixture(scope="session")
def example_regions(example_dataset_folder, example_transcriptome):
    biomart_dataset = chd.biomart.Dataset.from_genome("GRCh38")
    canonical_transcripts = chd.biomart.get_canonical_transcripts(
        biomart_dataset, example_transcriptome.var.index
    )
    regions = chd.data.Regions.from_canonical_transcripts(
        canonical_transcripts,
        path=example_dataset_folder / "regions",
        window=[-10000, 10000],
    )
    return regions


@pytest.fixture(scope="session")
def example_fragments(example_dataset_folder, example_transcriptome, example_regions):
    fragments = chd.data.Fragments.from_fragments_tsv(
        example_dataset_folder / "fragments.tsv.gz",
        example_regions,
        obs=example_transcriptome.obs,
        path=example_dataset_folder / "fragments",
    )
    fragments.create_cellxgene_indptr()
    return fragments
