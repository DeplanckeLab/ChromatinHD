import pybedtools
import pandas as pd
import numpy as np

import chromatinhd

import matplotlib as mpl
import pathlib

import polyptich

from chromatinhd.data.regions import center as center_peaks

def get_usecols_and_names(peakcaller):
    if peakcaller in ["macs2_leiden_0.1"]:
        usecols = [0, 1, 2, 6]
        names = ["chrom", "start", "end", "name"]
    else:
        usecols = [0, 1, 2]
        names = ["chrom", "start", "end"]
    return usecols, names


def extract_peaks(peaks_bed, promoter, peakcaller):
    if peaks_bed is None:
        return pd.DataFrame({"start": [], "end": [], "method": [], "peak": []})

    promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chrom", "start", "end"]])

    usecols, names = get_usecols_and_names(peakcaller)
    peaks = promoter_bed.intersect(peaks_bed, wb=True, nonamecheck=True).to_dataframe(usecols=usecols, names=names)

    # print(peaks)
    if peakcaller in ["macs2_leiden_0.1"]:
        peaks = peaks.rename(columns={"name": "cluster"})
        peaks["cluster"] = peaks["cluster"].astype(int)

    if peakcaller == "rolling_500":
        peaks["start"] = peaks["start"] + 250
        peaks["end"] = peaks["end"] + 250

    if len(peaks) > 0:
        peaks["peak"] = peaks["chrom"] + ":" + peaks["start"].astype(str) + "-" + peaks["end"].astype(str)
        peaks = center_peaks(peaks, promoter)
        peaks = peaks.set_index("peak")
    else:
        peaks = pd.DataFrame({"start": [], "end": [], "method": [], "peak": []})

    return peaks


class Peaks(polyptich.grid.Panel):
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
        fc="#555",
        lw=0.5,
    ):
        super().__init__((width, row_height * len(peakcallers) / 5))

        ax = self.ax
        ax.set_xlim(*window)
        for peakcaller, peaks_peakcaller in peaks.groupby("peakcaller"):
            y = peakcallers.index.get_loc(peakcaller)

            _plot_peaks(ax, peaks_peakcaller, y, lw=lw, fc=fc)
            if y > 0:
                ax.axhline(y, color="#DDD", zorder=10, lw=0.5)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_ylim(len(peakcallers), 0)
        if label_methods:
            ax.set_yticks(np.arange(len(peakcallers)) + 0.5)
            ax.set_yticks(np.arange(len(peakcallers) + 1), minor=True)
            ax.set_yticklabels(
                peakcallers["label"],
                fontsize=min(16 * row_height, 8),
                va="center",
                ha="right" if label_methods_side == "left" else "right",
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
            left=label_methods_side == "left",
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=1,
            pad=1,
            right=label_methods_side == "right",
            left=label_methods_side == "left",
        )
        if label_methods_side == "right":
            ax.yaxis.tick_right()

        ax.set_xticks([])

    @classmethod
    def from_bed(cls, region, peakcallers, window=None, **kwargs):
        peaks = _get_peaks(region, peakcallers)

        return cls(peaks, peakcallers, window=window, **kwargs)

    @classmethod
    def from_preloaded(cls, region, peakcallers, peakcaller_data, window=None, **kwargs):
        peaks = []
        for peakcaller_name, peakcaller in peakcallers.iterrows():
            data = peakcaller_data[peakcaller_name]
            peaks_peakcaller = (
                data.query("chrom == @region.chrom").query("start <= @region.end").query("end >= @region.start")
            ).assign(peakcaller=peakcaller_name)
            peaks_peakcaller["peak"] = (
                peaks_peakcaller["chrom"]
                + ":"
                + peaks_peakcaller["start"].astype(str)
                + "-"
                + peaks_peakcaller["end"].astype(str)
            )
            peaks.append(peaks_peakcaller.set_index(["peakcaller", "peak"]))
        peaks = pd.concat(peaks)
        if len(peaks) > 0:
            peaks = center_peaks(peaks, region)

        return cls(peaks, peakcallers, window=window, **kwargs)


class PeaksBroken(polyptich.grid.Broken):
    def __init__(
        self,
        peaks,
        peakcallers,
        breaking,
        label_methods=True,
        label_rows=True,
        label_methods_side="right",
        row_height=0.6,
        lw=0.5,
    ):
        super().__init__(breaking, height=row_height * len(peakcallers) / 5)

        # y axis
        if label_methods_side == "right":
            ax = self[0, -1]

            self[0, 0].set_yticks([])
        else:
            ax = self[0, 0]

            self[0, -1].set_yticks([])

        # label methods
        if label_methods:
            ax.set_yticks(np.arange(len(peakcallers)) + 0.5)
            ax.set_yticks(np.arange(len(peakcallers) + 1), minor=True)
            ax.set_yticklabels(
                peakcallers["label"],
                fontsize=min(16 * row_height, 10),
                va="center",
                ha="right" if label_methods_side == "left" else "right",
            )
        else:
            ax.set_yticks([])

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
        if label_methods_side == "right":
            ax.yaxis.tick_right()

        # label y
        panel, ax = self[0, 0]
        if label_rows is True:
            ax.set_ylabel("Putative\nCREs", rotation=0, ha="right", va="center")
        elif label_rows is not False:
            ax.set_ylabel(label_rows, rotation=0, ha="right", va="center")
        else:
            ax.set_ylabel("")

        # set ylim for each panel
        for i, (region, region_info), (panel, ax) in zip(
            range(len(breaking.regions)), breaking.regions.iterrows(), self
        ):
            ax.set_xticks([])
            ax.set_ylim(len(peakcallers), 0)

            for i in range(0, len(peakcallers)+1):
                ax.axhline(i, color="#DDD", zorder=10, lw=0.5, clip_on = False)

        # plot peaks per broken panel
        for peakcaller, peaks_peakcaller in peaks.groupby("peakcaller"):
            y = peakcallers.index.get_loc(peakcaller)
            for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), self):
                plotdata = peaks_peakcaller.loc[
                    ~(
                        (peaks_peakcaller["start"] > region_info["end"])
                        | (peaks_peakcaller["end"] < region_info["start"])
                    )
                ]

                ax.spines["bottom"].set_visible(False)

                _plot_peaks(ax, plotdata, y, lw=lw)

    @classmethod
    def from_bed(cls, region, peakcallers, breaking, **kwargs):
        """
        Plot peaks from bed files

        Parameters
        ----------
        region : pd.Series
            Genomic region to plot peaks in
        peakcallers : pd.DataFrame
            Peak calling methods to plot, with columns
            - label: label for the method
            - path: path to the bed file
        
        """
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


def _plot_peaks(ax, plotdata, y, lw=0.5, fc="#555"):
    if len(plotdata) == 0:
        return
    if ("cluster" not in plotdata.columns) or pd.isnull(plotdata["cluster"]).all():
        for _, peak in plotdata.iterrows():
            rect = mpl.patches.Rectangle(
                (peak["start"], y),
                peak["end"] - peak["start"],
                1,
                fc=fc,
                lw=0.5,
            )
            ax.add_patch(rect)
            # ax.plot([peak["start"]] * 2, [y, y + 1], color="grey", lw=lw)
            # ax.plot([peak["end"]] * 2, [y, y + 1], color="grey", lw=lw)
    else:
        n_clusters = plotdata["cluster"].max() + 1
        h = 1 / n_clusters
        for _, peak in plotdata.iterrows():
            rect = mpl.patches.Rectangle(
                (peak["start"], y + peak["cluster"] / n_clusters),
                peak["end"] - peak["start"],
                h,
                fc=fc,
                lw=0,
            )
            ax.add_patch(rect)
