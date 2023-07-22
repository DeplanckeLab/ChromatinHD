import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import (
    Flow,
    StoredTorchInt32,
    Stored,
    StoredTorchInt64,
    TSV,
    Linked,
)

from chromatinhd.data.regions import Regions

import torch
import math
import pathlib
import typing
import tqdm.auto as tqdm


class RawFragments:
    def __init__(self, file):
        self.file = file


class Fragments(Flow):
    """Fragments centered around a gene window"""

    regions = Linked("regions")
    """regions of the fragments"""

    coordinates = StoredTorchInt64("coordinates")
    """Coordinates of the fragments"""

    mapping = StoredTorchInt64("mapping")
    """Mapping of a fragment to a gene and a cell"""

    cellxgene_indptr = StoredTorchInt64("cellxgene_indptr")
    """Index pointers for each cellxgene combination"""

    regions = Linked("regions")

    def create_cellxgene_indptr(self):
        cellxgene = self.mapping[:, 0] * self.n_genes + self.mapping[:, 1]

        if not (cellxgene.diff() >= 0).all():
            raise ValueError(
                "Fragments should be ordered by cell then gene (ascending)"
            )

        n_cellxgene = self.n_genes * self.n_cells
        cellxgene_indptr = torch.nn.functional.pad(
            torch.cumsum(torch.bincount(cellxgene, minlength=n_cellxgene), 0), (1, 0)
        )
        assert self.coordinates.shape[0] == cellxgene_indptr[-1]
        if not (cellxgene_indptr.diff() >= 0).all():
            raise ValueError(
                "Fragments should be ordered by cell then gene (ascending)"
            )
        self.cellxgene_indptr = cellxgene_indptr

    _genemapping = None

    @property
    def genemapping(self):
        if self._genemapping is None:
            self._genemapping = self.mapping[:, 1].contiguous()
        return self._genemapping

    _cellmapping = None

    @property
    def cellmapping(self):
        if self._cellmapping is None:
            self._cellmapping = self.mapping[:, 0].contiguous()
        return self._cellmapping

    var = TSV("var")
    obs = TSV("obs")

    _n_genes = None

    @property
    def n_genes(self):
        if self._n_genes is None:
            self._n_genes = self.var.shape[0]
        return self._n_genes

    _n_cells = None

    @property
    def n_cells(self):
        if self._n_cells is None:
            self._n_cells = self.obs.shape[0]
        return self._n_cells

    @property
    def local_cellxgene_ix(self):
        return self.cellmapping * self.n_genes + self.genemapping

    def estimate_fragment_per_cellxgene(self):
        return math.ceil(self.coordinates.shape[0] / self.n_cells / self.n_genes * 2)

    def create_cut_data(self):
        cut_coordinates = self.coordinates.flatten()
        cut_coordinates = (cut_coordinates - self.window[0]) / (
            self.window[1] - self.window[0]
        )
        keep_cuts = (cut_coordinates >= 0) & (cut_coordinates <= 1)
        cut_coordinates = cut_coordinates[keep_cuts]

        self.cut_coordinates = cut_coordinates

        self.cut_local_gene_ix = self.genemapping.expand(2, -1).T.flatten()[keep_cuts]
        self.cut_local_cell_ix = self.cellmapping.expand(2, -1).T.flatten()[keep_cuts]

    @property
    def genes_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @classmethod
    def from_fragments_tsv(
        cls,
        fragments_file: typing.Union[pathlib.Path, str],
        regions: Regions,
        obs: pd.DataFrame,
        path: typing.Union[pathlib.Path, str],
        overwrite=True,
    ):
        """
        Create a Fragments object from a tsv file

        Parameters:
            fragments_file:
                fragments_file of the tsv file
            path:
                folder in which the fragments object will be created
            regions:
                regions object
        """

        if isinstance(fragments_file, str):
            fragments_file = pathlib.Path(fragments_file)
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not fragments_file.exists():
            raise FileNotFoundError(f"File {fragments_file} does not exist")
        if not overwrite and path.exists():
            raise FileExistsError(f"Folder {path} already exists")
        path.mkdir(parents=True, exist_ok=True)

        # region information
        var = pd.DataFrame(index=regions.coordinates.index)
        var["ix"] = np.arange(var.shape[0])

        n_genes = var.shape[0]

        # cell information
        obs["ix"] = np.arange(obs.shape[0])
        cell_to_cell_ix = obs["ix"].to_dict()

        n_cells = obs.shape[0]

        # load fragments tabix
        import pysam

        fragments_tabix = pysam.TabixFile(str(fragments_file))

        coordinates_raw = []
        mapping_raw = []

        for i, (gene, promoter_info) in tqdm.tqdm(
            enumerate(regions.coordinates.iterrows()),
            total=regions.coordinates.shape[0],
            leave=False,
            desc="Processing fragments",
        ):
            gene_ix = var.loc[gene, "ix"]
            start = max(0, promoter_info["start"])

            fragments_promoter = fragments_tabix.fetch(
                promoter_info["chrom"],
                start,
                promoter_info["end"],
                parser=pysam.asTuple(),
            )

            for fragment in fragments_promoter:
                cell = fragment[3]

                # only store the fragment if the cell is actually of interest
                if cell in cell_to_cell_ix:
                    # add raw data of fragment relative to tss
                    coordinates_raw.append(
                        [
                            (int(fragment[1]) - promoter_info["tss"])
                            * promoter_info["strand"],
                            (int(fragment[2]) - promoter_info["tss"])
                            * promoter_info["strand"],
                        ][:: promoter_info["strand"]]
                    )

                    # add mapping of cell/gene
                    mapping_raw.append([cell_to_cell_ix[fragment[3]], gene_ix])

        coordinates = torch.tensor(np.array(coordinates_raw, dtype=np.int64))
        mapping = torch.tensor(np.array(mapping_raw), dtype=torch.int64)

        # Sort `coordinates` and `mapping` according to `mapping`
        sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
        mapping = mapping[sorted_idx]
        coordinates = coordinates[sorted_idx]

        return cls.create(
            path=path,
            coordinates=coordinates,
            mapping=mapping,
            regions=regions,
            var=var,
            obs=obs,
        )


class ChunkedFragments(Flow):
    chunk_size = Stored("chunk_size")
    chunkcoords = StoredTorchInt64("chunkcoords")
    chunkcoords_indptr = StoredTorchInt32("chunkcoords_indptr")
    clusters = StoredTorchInt32("clusters")
    relcoords = StoredTorchInt32("relcoords")

    clusters = Stored("clusters")
    clusters_info = Stored("clusters_info")
    chromosomes = Stored("chromosomes")
