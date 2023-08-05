from __future__ import annotations
import math

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

from chromatinhd.data.regions import Regions
from chromatinhd.flow import TSV, Flow, Linked, Stored, StoredTensor, PathLike
from chromatinhd.utils import class_or_instancemethod

import pathlib


class RawFragments:
    def __init__(self, file):
        self.file = file


class Fragments(Flow):
    """Fragments centered around a gene window"""

    regions: Regions = Linked()
    """The regions where we stored fragments"""

    coordinates: torch.Tensor = StoredTensor(torch.int64)
    """Coordinates of the two cut sites."""

    mapping: torch.Tensor = StoredTensor(torch.int64)
    """Mapping of a fragment to a gene and a cell"""

    cellxgene_indptr: torch.Tensor = StoredTensor(torch.int64)
    """Index pointers for each cellxgene combination"""

    def create_cellxgene_indptr(self):
        cellxgene = self.mapping[:, 0] * self.n_genes + self.mapping[:, 1]

        if not (cellxgene.diff() >= 0).all():
            raise ValueError("Fragments should be ordered by cell then gene (ascending)")

        n_cellxgene = self.n_genes * self.n_cells
        cellxgene_indptr = torch.nn.functional.pad(
            torch.cumsum(torch.bincount(cellxgene, minlength=n_cellxgene), 0), (1, 0)
        )
        assert self.coordinates.shape[0] == cellxgene_indptr[-1]
        if not (cellxgene_indptr.diff() >= 0).all():
            raise ValueError("Fragments should be ordered by cell then gene (ascending)")
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

    var = TSV()
    """DataFrame containing information about regions."""

    obs = TSV()
    """DataFrame containing information about cells."""

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

    # def create_cut_data(self):
    #     cut_coordinates = self.coordinates.flatten()
    #     cut_coordinates = (cut_coordinates - self.window[0]) / (
    #         self.window[1] - self.window[0]
    #     )
    #     keep_cuts = (cut_coordinates >= 0) & (cut_coordinates <= 1)
    #     cut_coordinates = cut_coordinates[keep_cuts]

    #     self.cut_coordinates = cut_coordinates

    #     self.cut_local_gene_ix = self.genemapping.expand(2, -1).T.flatten()[keep_cuts]
    #     self.cut_local_cell_ix = self.cellmapping.expand(2, -1).T.flatten()[keep_cuts]

    @property
    def genes_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @class_or_instancemethod
    def from_fragments_tsv(
        cls,
        fragments_file: PathLike,
        regions: Regions,
        obs: pd.DataFrame,
        cell_column: str = None,
        path: PathLike = None,
        overwrite: bool = True,
    ) -> Fragments:
        """
        Create a Fragments object from a fragments tsv file

        Parameters:
            fragments_file:
                Location of the fragments tab-separate file created by e.g. CellRanger or sinto
            obs:
                DataFrame containing information about cells.
                The index should be the cell names as present in the fragments file.
                Alternatively, the column containing cell ids can be specified using the `cell_column` argument.
            regions:
                Regions from which the fragments will be extracted.
            cell_column:
                Column name in the `obs` DataFrame containing the cell names.
                If not specified, the index of the `obs` DataFrame is used.
            path:
                Folder in which the fragments data will be stored.
            overwrite:
                Whether to overwrite the data if it already exists.
        Returns:
            A new Fragments object
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

        # regions information
        var = pd.DataFrame(index=regions.coordinates.index)
        var["ix"] = np.arange(var.shape[0])

        # cell information
        obs = obs.copy()
        obs["ix"] = np.arange(obs.shape[0])
        if cell_column is None:
            cell_to_cell_ix = obs["ix"].to_dict()
        else:
            cell_to_cell_ix = obs.set_index(cell_column)["ix"].to_dict()

        # load fragments tabix
        import pysam

        fragments_tabix = pysam.TabixFile(str(fragments_file))

        coordinates_raw = []
        mapping_raw = []

        for _, (gene, promoter_info) in tqdm.tqdm(
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

            tss = promoter_info["tss"]
            strand = promoter_info["strand"]

            for fragment in fragments_promoter:
                cell = fragment[3]

                # only store the fragment if the cell is actually of interest
                if cell in cell_to_cell_ix:
                    # add raw data of fragment relative to tss
                    coordinates_raw.append(
                        [
                            (int(fragment[1]) - tss) * strand,
                            (int(fragment[2]) - tss) * strand,
                        ][::strand]
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

    def filter_genes(self, regions: Regions, path: PathLike = None) -> Fragments:
        """
        Filter based on new regions

        Parameters:
            regions:
                Regions to filter.
        Returns:
            A new Fragments object
        """

        # test if new regions are a subset of the existing ones
        if not regions.coordinates.index.isin(self.regions.coordinates.index).all():
            raise ValueError("New regions should be a subset of the existing ones")

        # filter genes
        self.regions.coordinates["ix"] = np.arange(self.regions.coordinates.shape[0])
        regions.coordinates["ix"] = self.regions.coordinates["ix"].loc[regions.coordinates.index]
        fragments_oi = np.isin(self.mapping[:, 1].numpy(), regions.coordinates["ix"])

        mapping = self.mapping[fragments_oi]
        coordinates = self.coordinates[fragments_oi]
        var = self.regions.coordinates.copy()
        var["original_ix"] = np.arange(var.shape[0])
        var = var.loc[regions.coordinates.index].copy()
        var["ix"] = np.arange(var.shape[0])
        mapping[:, 1] = torch.from_numpy(var.set_index("original_ix").loc[mapping[:, 1].cpu().numpy(), "ix"].values)

        # Sort `coordinates` and `mapping` according to `mapping`
        sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
        mapping = mapping[sorted_idx]
        coordinates = coordinates[sorted_idx]

        return Fragments.create(
            coordinates=coordinates, mapping=mapping, regions=regions, var=var, obs=self.obs, path=path
        )


class ChunkedFragments(Flow):
    chunk_size = Stored()
    chunkcoords = StoredTensor(dtype=torch.int64)
    chunkcoords_indptr = StoredTensor(dtype=torch.int32)
    clusters = StoredTensor(dtype=torch.int32)
    relcoords = StoredTensor(dtype=torch.int32)

    clusters = Stored()
    clusters_info = Stored()
    chromosomes = Stored()
