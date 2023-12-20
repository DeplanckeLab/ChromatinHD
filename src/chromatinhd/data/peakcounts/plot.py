import pybedtools
import pandas as pd
import numpy as np

import chromatinhd

import matplotlib as mpl
import pathlib


def center_peaks(peaks, promoter):
    peaks = peaks.copy()
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=["start", "end"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * int(promoter["strand"]),
                (peak["end"] - promoter["tss"]) * int(promoter["strand"]),
            ][:: int(promoter["strand"])]
            for _, peak in peaks.iterrows()
        ]
    return peaks


def uncenter_peaks(peaks, promoter):
    peaks = peaks.copy()
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=["start", "end"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] * int(promoter["strand"]) + promoter["tss"]),
                (peak["end"] * int(promoter["strand"]) + promoter["tss"]),
            ][:: int(promoter["strand"])]
            for _, peak in peaks.iterrows()
        ]
    return peaks


def get_usecols_and_names(peakcaller):
    if peakcaller in ["macs2_leiden_0.1"]:
        usecols = [0, 1, 2, 6]
        names = ["chrom", "start", "end", "name"]
    else:
        usecols = [0, 1, 2]
        names = ["chrom", "start", "end"]
    return usecols, names


def extract_peaks(peaks_bed, promoter, peakcaller):
    promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chrom", "start", "end"]])

    usecols, names = get_usecols_and_names(peakcaller)
    peaks = promoter_bed.intersect(peaks_bed, wb=True, nonamecheck=True).to_dataframe(usecols=usecols, names=names)

    if peakcaller in ["macs2_leiden_0.1"]:
        peaks = peaks.rename(columns={"name": "cluster"})
        peaks["cluster"] = peaks["cluster"].astype(int)

    if len(peaks) > 0:
        peaks["peak"] = peaks["chrom"] + ":" + peaks["start"].astype(str) + "-" + peaks["end"].astype(str)
        peaks = center_peaks(peaks, promoter)
        peaks = peaks.set_index("peak")
    else:
        peaks = pd.DataFrame({"start": [], "end": [], "method": [], "peak": []})
    return peaks


class Peaks(chromatinhd.grid.Ax):
    def __init__(
        self,
        peaks,
        peakcallers,
        window,
        width,
        label_methods=True,
        label_rows=True,
        label_methods_side="right",
        row_height=0.6,
    ):
        super().__init__((width, row_height * len(peakcallers) / 5))

        ax = self.ax
        ax.set_xlim(*window)
        for peakcaller, peaks_peakcaller in peaks.groupby("peakcaller"):
            y = peakcallers.index.get_loc(peakcaller)

            if len(peaks_peakcaller) == 0:
                continue
            if ("cluster" not in peaks_peakcaller.columns) or pd.isnull(peaks_peakcaller["cluster"]).all():
                for _, peak in peaks_peakcaller.iterrows():
                    rect = mpl.patches.Rectangle(
                        (peak["start"], y),
                        peak["end"] - peak["start"],
                        1,
                        fc="#333",
                        lw=0,
                    )
                    ax.add_patch(rect)
                    ax.plot([peak["start"]] * 2, [y, y + 1], color="grey", lw=0.5)
                    ax.plot([peak["end"]] * 2, [y, y + 1], color="grey", lw=0.5)
            else:
                n_clusters = peaks_peakcaller["cluster"].max() + 1
                h = 1 / n_clusters
                for _, peak in peaks_peakcaller.iterrows():
                    rect = mpl.patches.Rectangle(
                        (peak["start"], y + peak["cluster"] / n_clusters),
                        peak["end"] - peak["start"],
                        h,
                        fc="#333",
                        lw=0,
                    )
                    ax.add_patch(rect)
            if y > 0:
                ax.axhline(y, color="#DDD", zorder=10, lw=0.5)

        ax.set_ylim(len(peakcallers), 0)
        if label_methods:
            ax.set_yticks(np.arange(len(peakcallers)) + 0.5)
            ax.set_yticks(np.arange(len(peakcallers) + 1), minor=True)
            ax.set_yticklabels(
                peakcallers["label"],
                fontsize=min(16 * row_height, 10),
                va="center",
            )
        else:
            ax.set_yticks([])

        if label_rows is True:
            ax.set_ylabel("Putative\nCREs", rotation=0, ha="right", va="center")
        elif label_rows is not False:
            ax.set_ylabel(label_rows, rotation=0, ha="right", va="center")
        else:
            ax.set_ylabel("")
        ax.tick_params(
            axis="y",
            which="major",
            length=0,
            pad=1,
            right=label_methods_side == "right",
            left=not label_methods_side == "left",
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=1,
            pad=1,
            right=label_methods_side == "right",
            left=not label_methods_side == "left",
        )
        ax.yaxis.tick_right()

        ax.set_xticks([])

    @classmethod
    def from_bed(cls, region, peakcallers, **kwargs):
        peaks = _get_peaks(region, peakcallers)

        return cls(peaks, peakcallers, **kwargs)


class PeaksBroken(chromatinhd.grid.Broken):
    def __init__(
        self,
        peaks,
        peakcallers,
        breaking,
        label_methods=True,
        label_rows=True,
        label_methods_side="right",
        row_height=0.6,
    ):
        super().__init__(breaking, height=row_height * len(peakcallers) / 5)

        # y axis
        ax = self.elements[0, -1]

        ax.set_ylim(len(peakcallers), 0)
        if label_methods:
            ax.set_yticks(np.arange(len(peakcallers)) + 0.5)
            ax.set_yticks(np.arange(len(peakcallers) + 1), minor=True)
            ax.set_yticklabels(
                peakcallers["label"],
                fontsize=min(16 * row_height, 10),
                va="center",
            )
        else:
            ax.set_yticks([])

        if label_rows is True:
            ax.set_ylabel("Putative\nCREs", rotation=0, ha="right", va="center")
        elif label_rows is not False:
            ax.set_ylabel(label_rows, rotation=0, ha="right", va="center")
        else:
            ax.set_ylabel("")
        ax.tick_params(
            axis="y",
            which="major",
            length=0,
            pad=1,
            right=label_methods_side == "right",
            left=not label_methods_side == "left",
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=1,
            pad=1,
            right=label_methods_side == "right",
            left=not label_methods_side == "left",
        )
        ax.yaxis.tick_right()

        ax.set_xticks([])

        # plot peaks
        for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), self):
            for peakcaller, peaks_peakcaller in peaks.groupby("peakcaller"):
                y = peakcallers.index.get_loc(peakcaller)

                if len(peaks_peakcaller) == 0:
                    continue
                if ("cluster" not in peaks_peakcaller.columns) or pd.isnull(peaks_peakcaller["cluster"]).all():
                    for _, peak in peaks_peakcaller.iterrows():
                        rect = mpl.patches.Rectangle(
                            (peak["start"], y),
                            peak["end"] - peak["start"],
                            1,
                            fc="#333",
                            lw=0,
                        )
                        ax.add_patch(rect)
                        ax.plot([peak["start"]] * 2, [y, y + 1], color="grey", lw=0.5)
                        ax.plot([peak["end"]] * 2, [y, y + 1], color="grey", lw=0.5)
                else:
                    n_clusters = peaks_peakcaller["cluster"].max() + 1
                    h = 1 / n_clusters
                    for _, peak in peaks_peakcaller.iterrows():
                        rect = mpl.patches.Rectangle(
                            (peak["start"], y + peak["cluster"] / n_clusters),
                            peak["end"] - peak["start"],
                            h,
                            fc="#333",
                            lw=0,
                        )
                        ax.add_patch(rect)
                if y > 0:
                    ax.axhline(y, color="#DDD", zorder=10, lw=0.5)

    @classmethod
    def from_bed(cls, region, peakcallers, breaking, **kwargs):
        peaks = _get_peaks(region, peakcallers)

        return cls(peaks, peakcallers, breaking=breaking, **kwargs)


def _get_peaks(region, peakcallers):
    assert isinstance(region, pd.Series)
    assert isinstance(peakcallers, pd.DataFrame)
    assert "label" in peakcallers.columns
    assert "path" in peakcallers.columns

    peaks = []

    import pybedtools

    for peakcaller, peakcaller_info in peakcallers.iterrows():
        if not pathlib.Path(peakcaller_info["path"]).exists():
            continue
        peaks_bed = pybedtools.BedTool(str(pathlib.Path(peakcaller_info["path"])))

        peaks.append(extract_peaks(peaks_bed, region, peakcaller).assign(peakcaller=peakcaller))

    peaks = pd.concat(peaks).reset_index().set_index(["peakcaller", "peak"])
    peaks["size"] = peaks["end"] - peaks["start"]
    return peaks
