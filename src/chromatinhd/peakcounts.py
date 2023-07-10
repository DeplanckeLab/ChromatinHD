import numpy as np
import pandas as pd

import scanpy as sc

import collections
import subprocess as sp
import tqdm.auto as tqdm
import pathlib
import pickle

from chromatinhd.flow import Flow

htslib_folder = pathlib.Path("/data/peak_free_atac/software/htslib-1.16/")
tabix_location = htslib_folder / "tabix"


class PeakCounts(Flow):
    def create_adata(self, original_adata):
        adata = sc.AnnData(self.counts, obs=self.obs, var=self.var)
        adata.obsm["X_umap"] = original_adata.obsm["X_umap"]
        self.adata = adata

    def count_peaks(self, fragments_location, cell_ids):
        # extract unique peaks
        peaks = self.peaks
        unique_peak_ids = list(set(peaks["peak"]))
        peak_idxs = {peak_id: i for i, peak_id in enumerate(unique_peak_ids)}

        # create peaks file for tabix
        self.peaks_bed = peaks[["chrom", "start", "end"]]

        # count
        counts = collections.defaultdict(int)
        cell_ids = [str(cell_id) for cell_id in cell_ids]
        barcode_idxs = {barcode: ix for ix, barcode in enumerate(cell_ids)}

        process = sp.Popen(
            [
                tabix_location,
                fragments_location,
                "-R",
                self.peaks_bed_path,
                "--separate-regions",
            ],
            stdout=sp.PIPE,
        )
        counter = tqdm.tqdm(total=len(unique_peak_ids), smoothing=0)
        for line in process.stdout:
            line = line.decode("utf-8")
            if line.startswith("#"):
                peak = line.rstrip("\n").lstrip("#")
                peak_idx = peak_idxs[peak]
                counter.update(1)
            else:
                fragment = line.split("\t")
                barcode = fragment[3].strip("\n")

                if barcode in barcode_idxs:
                    counts[(barcode_idxs[barcode], peak_idx)] += 1

        # convert to sparse
        import scipy.sparse

        i = [k[0] for k in counts.keys()]
        j = [k[1] for k in counts.keys()]
        v = [v for v in counts.values()]
        counts_csr = scipy.sparse.csr_matrix(
            (v, (i, j)), shape=(len(barcode_idxs), len(peak_idxs))
        )

        if counts_csr.sum() == 0:
            raise ValueError("Something went wrong with counting")

        self.counts = counts_csr

        # create obs
        obs = pd.DataFrame(
            {"cell": list(barcode_idxs.keys()), "ix": list(barcode_idxs.values())}
        ).set_index("cell")
        self.obs = obs.copy()

        # create var
        var = pd.DataFrame(index=pd.Series(unique_peak_ids, name="peak"))
        var["ix"] = np.arange(var.shape[0])
        self.var = var

    @property
    def peaks_bed_path(self):
        return self.path / "peaks_bed.tsv"

    @property
    def peaks_bed(self):
        return pd.read_table(self.peaks_bed_path)

    @peaks_bed.setter
    def peaks_bed(self, value):
        value.to_csv(self.peaks_bed_path, sep="\t", header=False, index=False)

    @property
    def peaks(self):
        return pd.read_table(self.path / "peaks.tsv", index_col=0)

    @peaks.setter
    def peaks(self, value):
        value.index.name = "peak_gene"
        value.to_csv(self.path / "peaks.tsv", sep="\t")

    @property
    def counts(self):
        return pickle.load((self.path / "counts.pkl").open("rb"))

    @counts.setter
    def counts(self, value):
        pickle.dump(value, (self.path / "counts.pkl").open("wb"))

    @property
    def var(self):
        return pd.read_table(self.path / "var.tsv", index_col=0)

    @var.setter
    def var(self, value):
        value.index.name = "peak"
        value.to_csv(self.path / "var.tsv", sep="\t")

    @property
    def obs(self):
        return pd.read_table(self.path / "obs.tsv", index_col=0)

    @obs.setter
    def obs(self, value):
        value.index.name = "peak"
        value.to_csv(self.path / "obs.tsv", sep="\t")

    _adata = None

    @property
    def adata(self):
        if self._adata is None:
            self._adata = pickle.load((self.path / "adata.pkl").open("rb"))
        return self._adata

    @adata.setter
    def adata(self, value):
        pickle.dump(value, (self.path / "adata.pkl").open("wb"))
        self._adata = value


class FullPeak(PeakCounts):
    default_name = "full_peak"

    def create_peaks(self, original_peak_annot):
        original_peak_annot.index = pd.Index(
            original_peak_annot.chrom
            + ":"
            + original_peak_annot.start.astype(str)
            + "-"
            + original_peak_annot.end.astype(str)
        )

        peaks = original_peak_annot.copy()

        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str),
            name="peak",
        )
        peaks = peaks.groupby(level=0).first()
        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str),
            name="peak",
        )

        peaks["original_peak"] = peaks.index

        self.peaks = peaks


class HalfPeak(PeakCounts):
    default_name = "half_peak"

    def create_peaks(self, original_peak_annot):
        original_peak_annot.index = pd.Index(
            original_peak_annot.chrom
            + ":"
            + original_peak_annot.start.astype(str)
            + "-"
            + original_peak_annot.end.astype(str)
        )

        peaks = []
        for peak_id, peak in tqdm.tqdm(
            original_peak_annot.iterrows(), total=original_peak_annot.shape[0]
        ):
            peak1 = peak.copy()
            peak1["end"] = int(peak["start"] + int(peak["end"] - peak["start"]) / 2)
            peak1["original_peak"] = peak_id
            peaks.append(peak1)
            peak2 = peak.copy()
            peak2["start"] = int(peak["end"] - int(peak["end"] - peak["start"]) / 2)
            peak2["original_peak"] = peak_id
            peaks.append(peak2)

        peaks = pd.DataFrame(peaks)
        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
        )
        peaks = peaks.groupby(level=0).first()
        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
        )

        self.peaks = peaks


class ThirdPeak(PeakCounts):
    default_name = "third_peak"

    def create_peaks(self, original_peak_annot):
        original_peak_annot.index = pd.Index(
            original_peak_annot.chrom
            + ":"
            + original_peak_annot.start.astype(str)
            + "-"
            + original_peak_annot.end.astype(str)
        )

        peaks = []
        for peak_id, peak in original_peak_annot.iterrows():
            scale = peak["end"] - peak["start"]
            peak1 = peak.copy()
            peak1["end"] = int(peak["start"] + scale / 3)
            peak1["original_peak"] = peak_id
            peaks.append(peak1)
            peak2 = peak.copy()
            peak2["start"] = int(peak["start"] + scale / 3)
            peak2["end"] = int(peak["end"] - scale / 3)
            peak2["original_peak"] = peak_id
            peaks.append(peak2)
            peak2 = peak.copy()
            peak2["start"] = int(peak["end"] - scale / 3)
            peak2["original_peak"] = peak_id
            peaks.append(peak2)

        peaks = pd.DataFrame(peaks)
        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
        )
        peaks = peaks.groupby(level=0).first()
        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
        )

        self.peaks = peaks


class BroaderPeak(PeakCounts):
    default_name = "broader_peak"

    def create_peaks(self, original_peak_annot):
        original_peak_annot.index = pd.Index(
            original_peak_annot.chrom
            + ":"
            + original_peak_annot.start.astype(str)
            + "-"
            + original_peak_annot.end.astype(str)
        )

        peaks = original_peak_annot.copy()
        peaks["original_peak"] = peaks.index.copy()

        peaks["start"] = np.maximum(1, peaks["start"] - 5000)
        peaks["end"] = peaks["end"] + 5000

        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
        )
        peaks = peaks.groupby(level=0).first()
        peaks.index = pd.Index(
            peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str)
        )

        self.peaks = peaks


class FragmentPeak(FullPeak):
    default_name = "fragment_peak"

    def count_peaks(self, fragments_location, cell_ids):
        # add ix to peaks
        peaks = self.peaks
        peaks["ix"] = np.arange(peaks.shape[0])

        # create peaks file for tabix
        peaks_bed = peaks[["chrom", "start", "end"]]
        self.peaks_bed = peaks_bed

        # count
        fragments = []

        peak_idxs = peaks["ix"].to_dict()
        barcode_idxs = {barcode: ix for ix, barcode in enumerate(cell_ids)}

        fragment_cutoffs = (125, 250, 400)

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
        counter = tqdm.tqdm(total=self.peaks.shape[0], smoothing=0)
        for line in process.stdout:
            line = line.decode("utf-8")
            if line.startswith("#"):
                peak = line.rstrip("\n").lstrip("#")
                peak_idx = peak_idxs[peak]
                counter.update(1)
            else:
                fragment = line.split("\t")
                barcode = fragment[3]
                if barcode in barcode_idxs:
                    length = int(fragment[2]) - int(fragment[1])
                    for i, cutoff in enumerate(fragment_cutoffs):
                        if length < cutoff:
                            fragments.append((barcode_idxs[barcode], peak_idx, i))
                            break
                    else:
                        fragments.append((barcode_idxs[barcode], peak_idx, i))

            # if counter.n > 1000:
            #     break

        # count individual fragments
        counts = collections.defaultdict(int)
        for k in fragments:
            counts[(k[0], k[1] * (len(fragment_cutoffs) + 1) + k[2])] += 1

        # convert to sparse
        import scipy.sparse

        i = [k[0] for k in counts.keys()]
        j = [k[1] for k in counts.keys()]
        v = [v for v in counts.values()]
        counts_csr = scipy.sparse.csr_matrix(
            (v, (i, j)),
            shape=(len(barcode_idxs), len(peak_idxs) * (len(fragment_cutoffs) + 1)),
        )

        self.store("counts", counts_csr)

        # create obs
        obs = pd.DataFrame(
            {"cell": list(barcode_idxs.keys()), "ix": list(barcode_idxs.values())}
        ).set_index("cell")
        self.store("obs", obs)

        # create var
        var = peaks.loc[peaks.index.repeat(len(fragment_cutoffs) + 1)].copy()
        var["fragment_bin"] = (list(fragment_cutoffs) + ["inf"]) * peaks.shape[0]
        var.index = var.index + "_" + var["fragment_bin"].astype(str)

        self.var = var
