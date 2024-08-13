import numpy as np
import pandas as pd

import collections
import subprocess as sp
import tqdm.auto as tqdm
import pickle

from chromatinhd.flow import Flow, Stored, Linked

import tempfile
import pathlib


def count(peaks, tabix_location, fragments_location, barcode_idxs):
    # create peaks file for tabix
    folder = tempfile.TemporaryDirectory()
    peaks_bed_path = pathlib.Path(folder.name) / "peaks_bed.tsv"
    peaks[["chrom", "start", "end"]].to_csv(peaks_bed_path, sep="\t", header=False, index=False)
    peaks["start"] = np.clip(peaks["start"], 1, None)
    peaks.index = pd.Index(
        peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str),
        name="peak",
    )
    peak_idxs = {peak_id: i for i, peak_id in enumerate(peaks.index)}

    counts = collections.defaultdict(int)

    process = sp.Popen(
        [
            tabix_location,
            fragments_location,
            "-R",
            peaks_bed_path,
            "--separate-regions",
        ],
        stdout=sp.PIPE,
    )
    # counter = tqdm.tqdm(total=len(peak_idxs), smoothing=0)
    missing = 0
    for line in process.stdout:
        line = line.decode("utf-8")
        if line.startswith("#"):
            peak = line.rstrip("\n").lstrip("#")
            peak_idx = peak_idxs[peak]
            # counter.update(1)
        else:
            fragment = line.split("\t")
            barcode = fragment[3].strip("\n")

            if barcode in barcode_idxs:
                counts[(barcode_idxs[barcode], peak_idx)] += 1
            else:
                missing += 1

    # convert to sparse
    import scipy.sparse

    i = [k[0] for k in counts.keys()]
    j = [k[1] for k in counts.keys()]
    v = [v for v in counts.values()]
    counts_csr = scipy.sparse.csr_matrix((v, (i, j)), shape=(len(barcode_idxs), len(peak_idxs)))

    # if counts_csr.sum() == 0:
    #     raise ValueError("Something went wrong with counting")

    return counts_csr


class PeakCounts(Flow):
    cell_ids = Stored()

    tabix_location = Stored()
    fragments_location = Stored()

    fragments = Linked()

    def create_adata(self, original_adata):
        import scanpy as sc

        adata = sc.AnnData(self.counts, obs=self.obs, var=self.var)
        adata.obsm["X_umap"] = original_adata.obsm["X_umap"]
        self.adata = adata

    def count_peaks(self, fragments_location, cell_ids, tabix_location="tabix", do_count=True):
        self.fragments_location = fragments_location
        self.tabix_location = tabix_location

        peaks = self.peaks

        cell_ids = [str(cell_id) for cell_id in cell_ids]

        # create obs
        obs = pd.DataFrame({"cell": cell_ids, "ix": range(len(cell_ids))}).set_index("cell")
        self.obs = obs.copy()

        # create var
        var = peaks.groupby("peak").first()[["chrom", "start", "end"]]
        var["ix"] = np.arange(var.shape[0])
        self.var = var

        # do the counting
        if do_count:
            self.counts = self._count(var, tabix_location, fragments_location)

    def _count(self, peaks, tabix_location=None, fragments_location=None):
        if tabix_location is None:
            tabix_location = self.tabix_location
        if fragments_location is None:
            fragments_location = self.fragments_location
        barcode_idxs = self.obs["ix"].to_dict()

        return count(
            peaks, tabix_location=tabix_location, fragments_location=fragments_location, barcode_idxs=barcode_idxs
        )

    _counts_dense = None

    def get_peak_counts(self, region_oi, fragments_location=None, tabix_location=None, counts=None, densify=True):
        peak_gene_links_oi = self.peaks.loc[self.peaks["gene"] == region_oi].copy()
        peak_gene_links_oi["region"] = region_oi

        var_oi = self.var.loc[peak_gene_links_oi["peak"]]

        if (counts is None) and (self.o.counts.exists(self)):
            counts = self.counts
            if densify and (self._counts_dense is None):
                self._counts_dense = np.array(counts.todense())

        if self._counts_dense is not None:
            return peak_gene_links_oi, self._counts_dense[:, var_oi["ix"]]
        elif counts is not None:
            return peak_gene_links_oi, np.array(counts[:, var_oi["ix"]].todense())
        else:
            print("COUNTING!!")
            return peak_gene_links_oi, np.array(
                self._count(var_oi, tabix_location=tabix_location, fragments_location=fragments_location).todense()
            )

    def get_peaks_counts(self, regions_oi, fragments_location=None, tabix_location=None):
        peaks = []
        final_counts = []
        import scipy.sparse

        counts = scipy.sparse.csc_array(self.counts)
        for region_oi in tqdm.tqdm(regions_oi, leave=False):
            peaks_oi, counts_oi = self.get_peak_counts(
                region_oi, fragments_location=fragments_location, tabix_location=tabix_location, counts=counts
            )
            peaks.append(peaks_oi)
            final_counts.append(counts_oi)
        return pd.concat(peaks), np.concatenate(final_counts, axis=1)

    @property
    def peaks_bed_path(self):
        return self.path / "peaks_bed.tsv"

    @property
    def peaks_bed(self):
        return pd.read_table(self.peaks_bed_path)

    @peaks_bed.setter
    def peaks_bed(self, value):
        value.to_csv(self.peaks_bed_path, sep="\t", header=False, index=False)

    peaks = Stored()
    counts = Stored(compress=True)

    var = Stored()

    obs = Stored()
    adata = Stored()
    peaks = Stored()

    @property
    def counted(self):
        return self.o.counts.exists(self) and self.o.var.exists(self) and self.o.obs.exists(self)


class Windows(Flow):
    fragments = Linked()

    tabix_location = Stored()
    fragments_location = Stored()

    window_size = Stored()

    counted = True

    def get_peak_counts(self, region_oi, fragments_location=None, tabix_location=None, densify=None):
        region = self.fragments.regions.coordinates.loc[region_oi]
        starts = np.arange(region["start"], region["end"], step=self.window_size)
        ends = np.hstack([starts[1:], [region["end"]]])
        peaks = pd.DataFrame({"chrom": region["chrom"], "start": starts, "end": ends, "region": region_oi})

        peaks = center_peaks(peaks, region, columns=["relative_start", "relative_end"])

        barcode_idxs = self.fragments.obs["ix"].to_dict()

        counts = np.array(
            count(
                peaks,
                tabix_location=self.tabix_location,
                fragments_location=self.fragments_location,
                barcode_idxs=barcode_idxs,
            ).todense()
        )

        return peaks, counts

    def get_peaks_counts(self, regions_oi, fragments_location=None, tabix_location=None, densify=None):
        peaks = []
        counts = []
        for region_oi in tqdm.tqdm(regions_oi, leave=False):
            peaks_oi, counts_oi = self.get_peak_counts(
                region_oi, fragments_location=fragments_location, tabix_location=tabix_location
            )
            peaks.append(peaks_oi)
            counts.append(counts_oi)
        return pd.concat(peaks), np.concatenate(counts, axis=1)


def center_peaks(peaks, region, columns=["start", "end"]):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=[*columns])
    else:
        peaks[columns] = [
            [
                (peak["start"] - region["tss"]) * int(region["strand"]),
                (peak["end"] - region["tss"]) * int(region["strand"]),
            ][:: int(region["strand"])]
            for _, peak in peaks.iterrows()
        ]
    return peaks
