import laflow as laf

import numpy as np
import pandas as pd

import scanpy as sc

import collections
import subprocess as sp
import tqdm.auto as tqdm

import pathlib
htslib_folder = pathlib.Path("/home/wsaelens/projects/probabilistic-cell/mini/peakcheck/software/htslib-1.16/")
tabix_location = htslib_folder / "tabix"

class Dataset(laf.Flow):
    def create_adata(self, original_adata):
        adata = sc.AnnData(self.counts, obs = self.obs, var = self.var)
        adata.obsm["X_umap"] = original_adata.obsm["X_umap"]
        self.adata = adata

class PeakDataset(Dataset):
    def count_peaks(self, fragments_location, cell_ids):
        # add ix to peaks
        self.peaks["ix"] = np.arange(self.peaks.shape[0])

        # create peaks file for tabix
        self.peaks_bed = laf.objects.DataFrame(names = ["chrom", "start", "end"], extension = "tsv")
        peaks_bed = self.peaks[["chrom", "start", "end"]]
        self.peaks_bed = peaks_bed

        # count
        counts = collections.defaultdict(int)

        peak_idxs = self.peaks["ix"].to_dict()
        barcode_idxs = {barcode:ix for ix, barcode in enumerate(cell_ids)}

        process = sp.Popen([tabix_location, fragments_location, "-R", self.peaks_bed_.path, "--separate-regions"], stdout=sp.PIPE)
        counter = tqdm.tqdm(total = self.peaks.shape[0], smoothing = 0)
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
                    counts[(barcode_idxs[barcode], peak_idx)] += 1

        # convert to sparse
        import scipy.sparse
        i = [k[0] for k in counts.keys()]
        j = [k[1] for k in counts.keys()]
        v = [v for v in counts.values()]
        counts_csr = scipy.sparse.csr_matrix((v, (i, j)), shape = (len(barcode_idxs), len(peak_idxs)))

        self.store("counts", counts_csr)

        # create obs
        obs = pd.DataFrame({"cell":list(barcode_idxs.keys()), "ix":list(barcode_idxs.values())}).set_index("cell")
        self.store("obs", obs)

        # create var
        self.store("var", self.peaks.copy())

class FullPeak(PeakDataset):
    default_name = "full_peak"
    
    def create_peaks(self, original_peak_annot, gene_ids):
        # original_peak_annot = original_peak_annot.loc[original_peak_annot["gene"].isin(gene_ids)]
        original_peak_annot.index = pd.Index(original_peak_annot.chrom + ":" + original_peak_annot.start.astype(str) + "-" + original_peak_annot.end.astype(str))

        peaks = original_peak_annot.copy()

        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str), name = "peak")
        peaks = peaks.groupby(level = 0).first()
        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str), name = "peak")

        peaks["original_peak"] = peaks.index

        self.peaks = peaks


class HalfPeak(PeakDataset):
    default_name = "half_peak"
    def create_peaks(self, original_peak_annot, gene_ids):
        original_peak_annot = original_peak_annot.loc[original_peak_annot["gene"].isin(gene_ids)]
        original_peak_annot.index = pd.Index(original_peak_annot.chrom + ":" + original_peak_annot.start.astype(str) + "-" + original_peak_annot.end.astype(str))

        peaks = []
        for peak_id, peak in original_peak_annot.iterrows():
            peak1 = peak.copy()
            peak1["end"] = int(peak["start"] + int(peak["end"] - peak["start"])/2)
            peak1["original_peak"] = peak_id
            peaks.append(peak1)
            peak2 = peak.copy()
            peak2["start"] = int(peak["end"] - int(peak["end"] - peak["start"])/2)
            peak2["original_peak"] = peak_id
            peaks.append(peak2)

        peaks = pd.DataFrame(peaks)
        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str))
        peaks = peaks.groupby(level = 0).first()
        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str))

        self.peaks = peaks


class ThirdPeak(PeakDataset):
    default_name = "third_peak"
    def create_peaks(self, original_peak_annot, gene_ids):
        original_peak_annot = original_peak_annot.loc[original_peak_annot["gene"].isin(gene_ids)]
        original_peak_annot.index = pd.Index(original_peak_annot.chrom + ":" + original_peak_annot.start.astype(str) + "-" + original_peak_annot.end.astype(str))

        peaks = []
        for peak_id, peak in original_peak_annot.iterrows():
            scale = peak["end"] - peak["start"]
            peak1 = peak.copy()
            peak1["end"] = int(peak["start"] + scale/3)
            peak1["original_peak"] = peak_id
            peaks.append(peak1)
            peak2 = peak.copy()
            peak2["start"] = int(peak["start"]+ scale/3)
            peak2["end"] = int(peak["end"]- scale/3)
            peak2["original_peak"] = peak_id
            peaks.append(peak2)
            peak2 = peak.copy()
            peak2["start"] = int(peak["end"] - scale/3)
            peak2["original_peak"] = peak_id
            peaks.append(peak2)

        peaks = pd.DataFrame(peaks)
        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str))
        peaks = peaks.groupby(level = 0).first()
        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str))

        self.peaks = peaks


class BroaderPeak(PeakDataset):
    default_name = "broader_peak"
    def create_peaks(self, original_peak_annot, gene_ids):
        original_peak_annot = original_peak_annot.loc[original_peak_annot["gene"].isin(gene_ids)]
        original_peak_annot.index = pd.Index(original_peak_annot.chrom + ":" + original_peak_annot.start.astype(str) + "-" + original_peak_annot.end.astype(str))

        peaks = original_peak_annot.copy()
        peaks["original_peak"] = peaks.index.copy()

        peaks["start"] = np.maximum(1, peaks["start"] - 5000)
        peaks["end"] = peaks["end"] + 5000

        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str))
        peaks = peaks.groupby(level = 0).first()
        peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str))

        self.peaks = peaks



class FragmentPeak(FullPeak):
    default_name = "fragment_peak"
    def count_peaks(self, fragments_location, cell_ids):
        # add ix to peaks
        self.peaks["ix"] = np.arange(self.peaks.shape[0])

        # create peaks file for tabix
        self.peaks_bed = laf.objects.DataFrame(names = ["chrom", "start", "end"], extension = "tsv")
        peaks_bed = self.peaks[["chrom", "start", "end"]]
        self.peaks_bed = peaks_bed

        # count
        fragments = []

        peak_idxs = self.peaks["ix"].to_dict()
        barcode_idxs = {barcode:ix for ix, barcode in enumerate(cell_ids)}

        fragment_cutoffs = (125, 250, 400)

        process = sp.Popen([tabix_location, fragments_location, "-R", self.peaks_bed_.path, "--separate-regions"], stdout=sp.PIPE)
        counter = tqdm.tqdm(total = self.peaks.shape[0], smoothing = 0)
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
            counts[(k[0], k[1]*(len(fragment_cutoffs) + 1) + k[2])] += 1

        # convert to sparse
        import scipy.sparse
        i = [k[0] for k in counts.keys()]
        j = [k[1] for k in counts.keys()]
        v = [v for v in counts.values()]
        counts_csr = scipy.sparse.csr_matrix((v, (i, j)), shape = (len(barcode_idxs), len(peak_idxs) * (len(fragment_cutoffs) + 1)))

        self.store("counts", counts_csr)

        # create obs
        obs = pd.DataFrame({"cell":list(barcode_idxs.keys()), "ix":list(barcode_idxs.values())}).set_index("cell")
        self.store("obs", obs)

        # create var
        var = self.peaks.loc[self.peaks.index.repeat(len(fragment_cutoffs) + 1)].copy()
        var["fragment_bin"] = (list(fragment_cutoffs) + ["inf"]) * self.peaks.shape[0]
        var.index = var.index + "_" + var["fragment_bin"].astype(str)
        self.store("var", var)