import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

import chromatinhd
import chromatinhd as chd
import polyptich
from polyptich.grid.broken import Broken
from chromatinhd.plot import gene_ticker, format_distance, round_significant



def center(coords: pd.DataFrame, region: pd.Series, window: np.ndarray = None):
    """
    Center coordinates around a region

    Parameters:
        coords: The coordinates, with columns start and end
        region: The region, with columns start, end, and strand
        window: The window to filter to
    """
    coords = coords.copy()
    if coords.shape[0] == 0:
        coords = pd.DataFrame(columns=["start", "end", "method"])
    else:
        coords[["start", "end"]] = [
            [
                (peak["start"] - region["tss"]) * region["strand"],
                (peak["end"] - region["tss"]) * region["strand"],
            ][:: int(region["strand"])]
            for _, peak in coords.iterrows()
        ]

    if window is not None:
        # partial overlap
        coords = coords.loc[~((coords["end"] < window[0]) | (coords["start"] > window[1]))]
    return coords


def get_genes_plotdata(region, genome="GRCh38", window=None, use_cache=True, only_canonical=True):
    biomart_dataset = chd.biomart.Dataset.from_genome(genome)
    if "chrom" not in region.index:
        region = region.copy()
        region["chrom"] = region["chr"]
    if only_canonical:
        transcripts = chd.biomart.get_canonical_transcripts(
            biomart_dataset, chrom=region["chrom"], start=region["start"], end=region["end"], use_cache=use_cache
        )
    else:
        transcripts = chd.biomart.get_transcripts(
            biomart_dataset,
            chrom=region["chrom"],
            start=region["start"],
            end=region["end"],
        )
    exons = chd.biomart.get_exons(biomart_dataset, chrom=region["chrom"], start=region["start"], end=region["end"])

    plotdata_genes = transcripts
    plotdata_genes = center(plotdata_genes, region, window=window)

    plotdata_exons = exons.rename(
        columns={
            "exon_chrom_start": "start",
            "exon_chrom_end": "end",
            "ensembl_gene_id": "gene",
            "ensembl_transcript_id": "transcript",
        }
    )
    plotdata_exons = center(plotdata_exons, region, window=window)

    plotdata_coding = exons.dropna().rename(
        columns={
            "genomic_coding_start": "start",
            "genomic_coding_end": "end",
            "ensembl_gene_id": "gene",
            "ensembl_transcript_id": "transcript",
        }
    )
    plotdata_coding = center(plotdata_coding, region, window=window)

    return plotdata_genes, plotdata_exons, plotdata_coding


class Genes(polyptich.grid.Panel):
    def __init__(
        self,
        plotdata_genes,
        plotdata_exons,
        plotdata_coding,
        gene_id,
        region,
        window,
        width,
        full_ticks=False,
        label_genome=False,
        annotate_tss=True,
        symbol=None,
        **kwargs,
    ):
        super().__init__((width, len(plotdata_genes) * 0.08 + 0.01), **kwargs)

        ax = self.ax

        ax.xaxis.tick_top()
        if label_genome:
            if symbol is None:
                symbol = gene_id
            ax.set_xlabel("Distance to $\\mathit{" + symbol + "}$ TSS")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", length=2, pad=0, labelsize=8, width=0.5)
        ax.xaxis.set_major_formatter(chromatinhd.plot.gene_ticker)

        sns.despine(ax=ax, right=True, left=True, bottom=True, top=True)

        ax.set_xlim(*window)

        ax.set_yticks([])
        ax.set_ylabel("")

        if len(plotdata_genes) == 0:
            return

        plotdata_genes["ix"] = np.arange(len(plotdata_genes))

        ax.set_ylim(-0.5, plotdata_genes["ix"].max() + 0.5)
        if full_ticks:
            ax.set_xticks(np.arange(window[0], window[1] + 1, 500))
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.tick_params(
                axis="x",
                length=2,
                pad=0,
                labelsize=8,
                width=0.5,
                labelrotation=90,
            )
            ax.get_xticklabels()[0].set_horizontalalignment("left")
            ax.get_xticklabels()[-1].set_horizontalalignment("right")
        for gene, gene_info in plotdata_genes.reset_index().set_index("transcript").iterrows():
            y = gene_info["ix"]
            is_oi = gene == gene_id
            ax.plot(
                [gene_info["start"], gene_info["end"]],
                [y, y],
                color="black" if is_oi else "grey",
            )

            if pd.isnull(gene_info["symbol"]):
                symbol = gene_info.name
            else:
                symbol = gene_info["symbol"]
            strand = gene_info["strand"] * region["strand"]
            if (gene_info["start"] > window[0]) & (gene_info["start"] < window[1]):
                label = symbol + " → " if strand == 1 else "← " + symbol + " "
                ha = "right"
                # ha = "left" if (strand == -1) else "right"

                ax.text(
                    gene_info["start"],
                    y,
                    label,
                    style="italic",
                    ha=ha,
                    va="center",
                    fontsize=6,
                    weight="bold" if is_oi else "regular",
                )
            elif (gene_info["end"] > window[0]) & (gene_info["end"] < window[1]):
                label = " → " + symbol if strand == 1 else symbol + " ← "
                ha = "left"

                ax.text(
                    gene_info["end"],
                    y,
                    label,
                    style="italic",
                    ha=ha,
                    va="center",
                    fontsize=6,
                    weight="bold" if is_oi else "regular",
                )
            else:
                ax.text(
                    window[0] + (window[1] - window[0]) / 2,
                    y,
                    "(" + symbol + ")",
                    style="italic",
                    ha="center",
                    va="center",
                    fontsize=6,
                    bbox=dict(facecolor="#FFFFFF", boxstyle="square,pad=0", lw=0),
                )

            plotdata_exons_gene = plotdata_exons.query("transcript == @gene")
            h = 1
            for exon, exon_info in plotdata_exons_gene.iterrows():
                rect = mpl.patches.Rectangle(
                    (exon_info["start"], y - h / 2),
                    exon_info["end"] - exon_info["start"],
                    h,
                    fc="white",
                    ec="#333333",
                    lw=1.0,
                    zorder=9,
                )
                ax.add_patch(rect)

            plotdata_coding_gene = plotdata_coding.query("transcript == @gene")
            for coding, coding_info in plotdata_coding_gene.iterrows():
                rect = mpl.patches.Rectangle(
                    (coding_info["start"], y - h / 2),
                    coding_info["end"] - coding_info["start"],
                    h,
                    fc="#333333",
                    ec="#333333",
                    lw=1.0,
                    zorder=10,
                )
                ax.add_patch(rect)

        # vline at tss
        if annotate_tss:
            ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

    @classmethod
    def from_region(
        cls, region, genome="GRCh38", window=None, use_cache=True, show_genes=True, only_canonical=True, **kwargs
    ):
        if window is None:
            assert "tss" in region
            window = np.array([region["start"] - region["tss"], region["end"] - region["tss"]])
        plotdata_genes, plotdata_exons, plotdata_coding = get_genes_plotdata(
            region, genome=genome, window=window, use_cache=use_cache, only_canonical=only_canonical
        )

        if not show_genes:
            plotdata_genes = plotdata_genes.iloc[0:0]
            plotdata_exons = plotdata_exons.iloc[0:0]
            plotdata_coding = plotdata_coding.iloc[0:0]

        return cls(
            plotdata_genes=plotdata_genes,
            plotdata_exons=plotdata_exons,
            plotdata_coding=plotdata_coding,
            region=region,
            gene_id=region.name,
            window=window,
            **kwargs,
        )


def filter_start_end(x, start, end):
    y = x.loc[~((x["end"] < start) | (x["start"] > end))]
    return y


def filter_position(x, start, end):
    y = x.loc[~((x["position"] < start) | (x["position"] > end))]
    return y


class GenesBroken(Broken):
    def __init__(
        self,
        plotdata_genes,
        plotdata_exons,
        plotdata_coding,
        breaking,
        gene_id,
        *args,
        label_positions=True,
        label_positions_minlength=500,
        label_positions_rotation=0,
        **kwargs,
    ):

        height = len(plotdata_genes) * 0.08
        super().__init__(
            breaking=breaking,
            height=height,
            *args,
            **kwargs,
        )

        if len(plotdata_genes) == 0:
            return

        plotdata_genes["ix"] = np.arange(len(plotdata_genes))

        ylim = (-0.5, plotdata_genes["ix"].max() + 0.5)

        for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), self):
            # prepare axis
            ax.xaxis.tick_top()
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="x", length=2, pad=0, labelsize=8, width=0.5)
            ax.xaxis.set_major_formatter(gene_ticker)

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_xlim(region_info["start"], region_info["end"])
            ax.set_ylim(*ylim)

            # add label of position
            if label_positions:
                y = ylim[1] + 1.0
                if region_info["end"] - region_info["start"] > label_positions_minlength:
                    y_text = y
                    if label_positions_rotation == 90:
                        y_text = y_text + 0.5
                    ax.annotate(
                        format_distance(
                            round_significant(
                                int(region_info["start"] + (region_info["end"] - region_info["start"]) / 2), 2
                            ),
                            None,
                        ),
                        (0.5, y_text),
                        xycoords=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                        ha="center",
                        va="center" if label_positions_rotation == 0 else "bottom",
                        rotation=label_positions_rotation,
                        fontsize=6,
                        color="#999999",
                        bbox=dict(facecolor="#FFFFFF", boxstyle="square,pad=0.1", lw=0),
                    )
                line = mpl.lines.Line2D(
                    [0.0, 1.0],
                    [y, y],
                    transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    color="#999999",
                    lw=0.5,
                    clip_on=False,
                )
                ax.add_line(line)

            # plot genes
            plotdata_genes_region = filter_start_end(plotdata_genes, region_info["start"], region_info["end"])

            for transcript, gene_info in plotdata_genes_region.reset_index().set_index("transcript").iterrows():
                y = gene_info["ix"]
                is_oi = gene_info["ensembl_gene_id"] == gene_id
                
                if is_oi:
                    color = "#000000"
                else:
                    color = "#888888"
                ax.plot(
                    [gene_info["start"], gene_info["end"]],
                    [y, y],
                    color=color,
                )

                plotdata_exons_gene = plotdata_exons.query("transcript == @transcript")
                plotdata_exons_gene = filter_start_end(plotdata_exons_gene, region_info["start"], region_info["end"])
                h = 1
                for exon, exon_info in plotdata_exons_gene.iterrows():
                    rect = mpl.patches.Rectangle(
                        (exon_info["start"], y - h / 2),
                        exon_info["end"] - exon_info["start"],
                        h,
                        fc="white",
                        ec=color,
                        lw=1.0,
                        zorder=9,
                    )
                    ax.add_patch(rect)

                plotdata_coding_gene = plotdata_coding.query("transcript == @transcript")
                plotdata_coding_gene = filter_start_end(plotdata_coding_gene, region_info["start"], region_info["end"])
                for coding, coding_info in plotdata_coding_gene.iterrows():
                    rect = mpl.patches.Rectangle(
                        (coding_info["start"], y - h / 2),
                        coding_info["end"] - coding_info["start"],
                        h,
                        fc=color,
                        ec=color,
                        lw=1.0,
                        zorder=10,
                    )
                    ax.add_patch(rect)

            # vline at tss
            ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax = self[0, 0]
        ax.set_yticks(np.arange(len(plotdata_genes)))
        ax.set_yticklabels(plotdata_genes["symbol"], fontsize=6, style="italic")
        for tick in ax.yaxis.get_major_ticks():
            if tick.label1.get_text() == plotdata_genes.loc[gene_id, "symbol"]:
                tick.label1.set_weight("bold")
        ax.tick_params(axis="y", length=0, pad=2, width=0.5)

    @classmethod
    def from_region(cls, region, breaking, genome="GRCh38", use_cache=True, only_canonical=True, show_others = True, **kwargs):
        plotdata_genes, plotdata_exons, plotdata_coding = get_genes_plotdata(
            region, genome=genome, use_cache=use_cache, only_canonical=only_canonical
        )

        if not show_others:
            plotdata_genes = plotdata_genes.query("ensembl_gene_id == @region.name").copy()
            plotdata_exons = plotdata_exons.query("gene == @region.name").copy()
            plotdata_coding = plotdata_coding.query("gene == @region.name").copy()

        return cls(
            plotdata_genes=plotdata_genes,
            plotdata_exons=plotdata_exons,
            plotdata_coding=plotdata_coding,
            gene_id=region.name,
            breaking=breaking,
            **kwargs,
        )





class GenesExpanding(polyptich.grid.Panel):
    """
    Shows all genes in the regions, with a "zoom-in" effect towards the regions of interest.

    Parameters:
        plotdata_genes: pd.DataFrame
            DataFrame with gene information
        plotdata_exons: pd.DataFrame
            DataFrame with exon information
        plotdata_coding: pd.DataFrame
            DataFrame with coding information
        gene_id: str
            Gene ID of the region of interest
        region: pd.Series
            Region information
        breaking: polyptich.grid.Broken
            Broken grid
        window: np.ndarray
            Window to show
        full_ticks: bool
            Show full ticks
        label_genome: bool
            Label the genome
        annotate_tss: bool
            Annotate the TSS
        symbol: str
            Symbol of the region of interest
        expansion_height: float
            Height of the expansion
        xticks: list
            X-ticks

    
    """
    def __init__(
        self,
        plotdata_genes,
        plotdata_exons,
        plotdata_coding,
        gene_id,
        region,
        breaking,
        window,
        full_ticks=False,
        label_genome=False,
        annotate_tss=True,
        symbol=None,
        expansion_height = 2.5,
        xticks = None,
        gene_overlap_padding = 10000,
        **kwargs,
    ):
        width = breaking.width
        
        super().__init__((width, (2+expansion_height+len(plotdata_genes)) * 0.08 + 0.01), **kwargs)

        ax = self

        ax.xaxis.tick_top()
        if label_genome:
            if symbol is None:
                symbol = gene_id
            ax.set_xlabel("Distance to $\\mathit{" + symbol + "}$ TSS")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", length=2, pad=0, labelsize=8, width=0.5)
        if xticks is not None:
            if xticks == "extent":
                xticks = [window[0], window[1]]
            else:
                xticks = np.array(xticks)
            ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(chromatinhd.plot.gene_ticker)

        sns.despine(ax=ax, right=True, left=True, bottom=True, top=True)

        # top bar
        ax.axhline(0.5, color = "#333", lw = 1., zorder = -10)

        ax.set_xlim(*window)

        ax.set_yticks([])
        ax.set_ylabel("")

        if len(plotdata_genes) == 0:
            return
        
        # determine y-value of gene
        # we also add some gene_overlap_padding so that genes that non-overlapping genes are not too close
        plotdata_genes = plotdata_genes.sort_values("start")
        plotdata_genes["ix"] = 0.
        for i, (gene, gene_info) in enumerate(plotdata_genes.iterrows()):
            prev_genes = plotdata_genes.iloc[0:i]
            possible_ys = prev_genes["ix"].values.tolist() + [0, prev_genes["ix"].max() + 1]
            overlapping_prev_genes = prev_genes.query("end+@gene_overlap_padding > @gene_info['start']")
            possible_ys = [y for y in possible_ys if y not in overlapping_prev_genes["ix"].values]
            y = min(possible_ys)
            plotdata_genes.loc[gene, "ix"] = y
        
        self.height = (2+expansion_height+plotdata_genes["ix"].max()) * 0.08 + 0.01

        genes_max = plotdata_genes["ix"].max() + 0.5
        ax.set_ylim(-0.5 - expansion_height, genes_max)
        if full_ticks:
            ax.set_xticks(np.arange(window[0], window[1] + 1, 500))
            ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.tick_params(
                axis="x",
                length=2,
                pad=0,
                labelsize=8,
                width=0.5,
                labelrotation=90,
            )

        for gene, gene_info in plotdata_genes.reset_index().set_index("transcript").iterrows():
            y = gene_info["ix"]
            is_oi = (gene == gene_id) or ("gene" in gene_info and gene_info["gene"] == gene_id)
            ax.plot(
                [gene_info["start"], gene_info["end"]],
                [y, y],
                color="black" if is_oi else "grey",
            )

            if pd.isnull(gene_info["symbol"]):
                symbol = gene_info.name
            else:
                symbol = gene_info["symbol"]
            strand = gene_info["strand"] * region["strand"]
            if (gene_info["start"] > window[0]) & (gene_info["start"] < window[1]):
                label = symbol + " → " if strand == 1 else "← " + symbol + " "
                ha = "right"
                # ha = "left" if (strand == -1) else "right"

                text = ax.text(
                    gene_info["start"],
                    y,
                    label,
                    style="italic",
                    ha=ha,
                    va="center",
                    fontsize=6,
                    weight="bold" if is_oi else "regular",
                    zorder = 20,
                )
            elif (gene_info["end"] > window[0]) & (gene_info["end"] < window[1]):
                label = " → " + symbol if strand == 1 else symbol + " ← "
                ha = "left"

                text = ax.text(
                    gene_info["end"],
                    y,
                    label,
                    style="italic",
                    ha=ha,
                    va="center",
                    fontsize=6,
                    weight="bold" if is_oi else "regular",
                    zorder = 20,
                )
            else:
                text = ax.text(
                    window[0] + (window[1] - window[0]) / 2,
                    y,
                    "(" + symbol + ")",
                    style="italic",
                    ha="center",
                    va="center",
                    fontsize=6,
                    bbox=dict(facecolor="#FFFFFF", boxstyle="square,pad=0", lw=0),
                    zorder = 20,
                )
            text.set_path_effects(
                [
                    mpl.patheffects.withStroke(linewidth=2, foreground="#FFFFFF"),
                ]
            )

            plotdata_exons_gene = plotdata_exons.query("transcript == @gene")
            h = 1
            for exon, exon_info in plotdata_exons_gene.iterrows():
                rect = mpl.patches.Rectangle(
                    (exon_info["start"], y - h / 2),
                    exon_info["end"] - exon_info["start"],
                    h,
                    fc="white",
                    ec="#333333",
                    lw=1.0,
                    zorder=9,
                )
                ax.add_patch(rect)

            plotdata_coding_gene = plotdata_coding.query("transcript == @gene")
            for coding, coding_info in plotdata_coding_gene.iterrows():
                rect = mpl.patches.Rectangle(
                    (coding_info["start"], y - h / 2),
                    coding_info["end"] - coding_info["start"],
                    h,
                    fc="#333333",
                    ec="#333333",
                    lw=1.0,
                    zorder=10,
                )
                ax.add_patch(rect)

        # vline at tss
        if annotate_tss:
            ax.plot([0, 0], [0, genes_max], color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        # breaking
        from polyptich.grid.broken import TransformBroken
        breaking_transform = TransformBroken(breaking)
        max_breaking_transform = breaking_transform(breaking.regions["end"].max())[0]
        for i, (region, region_info) in enumerate(breaking.regions.iterrows()):
            end_broken = (breaking_transform(region_info["end"])[0] / max_breaking_transform * (window[1] - window[0])) + window[0]
            start_broken = (breaking_transform(region_info["start"])[0] / max_breaking_transform * (window[1] - window[0])) + window[0]

            control_point_height = expansion_height*0.5

            points = np.array([
                [region_info["start"], genes_max], # top-left
                [region_info["end"], genes_max], # top right
                [region_info["end"], -0.5], # bottom right
                [region_info["end"], -0.5-control_point_height], # bottom right P1
                [end_broken, -0.5-control_point_height], # bottom right P2
                [end_broken, -0.5-expansion_height], # bottom right
                [start_broken,-0.5-expansion_height],   # bottom left zoom
                [start_broken, -0.5-control_point_height], # bottom left P1
                [region_info["start"], -0.5-control_point_height], # bottom left P2
                [region_info["start"], -0.5], # bottom left
                [region_info["start"], genes_max], # top left
            ])

            # polygon
            # polygon = mpl.patches.Polygon(
            #     points,
            #     fc="#CCCCCC",
            #     lw=0.,
            #     zorder=-2,
            #     clip_on = False
            # )

            # smooth bezier path
            Path = mpl.path.Path

            path = mpl.path.Path(points, codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.LINETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CLOSEPOLY
            ])
            
            polygon = mpl.patches.PathPatch(
                path,
                fc="#D6D6D6" if i % 2 == 0 else "#E6E6E6",
                lw=0.,
                zorder=-2,
                clip_on = False
            )
            ax.add_patch(polygon)

            # ax.plot(
            #     [region_info["start"], region_info["start"], start_broken],
            #     [genes_max, -0.5, -0.5-expansion_height],
            #     color="#AAAAAA",
            #     lw=0.5,
            #     zorder=-1,
            # )
            # ax.plot(
            #     [region_info["end"], region_info["end"], end_broken],
            #     [genes_max, -0.5, -0.5-expansion_height],
            #     color="#AAAAAA",
            #     lw=0.5,
            #     zorder=-1,
            # )

            if (region_info["start"] < 0) and (region_info["end"] > 0):
                x = (breaking_transform(0)[0] / max_breaking_transform * (window[1] - window[0])) + window[0]
                path = mpl.path.Path(
                    [
                        [0, genes_max],
                        [0, -0.5],
                        [0, -0.5-control_point_height],
                        [x, -0.5-control_point_height],
                        [x, -0.5-expansion_height],
                    ],
                    [
                        Path.MOVETO,
                        Path.LINETO,
                        Path.CURVE4,
                        Path.CURVE4,
                        Path.CURVE4,
                    ]
                )
                polygon = mpl.patches.PathPatch(
                    path,
                    fc = "#FFFFFF00",
                    ec = "#AAAAAA",
                    lw=1,
                    linestyle="dotted",
                    zorder=-1,
                    clip_on = False
                )
                ax.add_patch(polygon)
                        


    @classmethod
    def from_region(
        cls, region:pd.Series, breaking, genome:str="GRCh38", window=None, use_cache=True, only_canonical=True, show_others = True, **kwargs
    ):
        """
        Create a GenesExpanding panel from a region.

        Parameters:
            region:
                Region information
            breaking:
                Broken grid
            genome:
                Genome
            window:
                Window to show
            use_cache:
                Use the cache for biomart
            only_canonical: bool
                Only show canonical transcripts
            show_others: bool
                Show other genes
        
        """
        if window is None:
            assert "tss" in region
            window = np.array([region["start"] - region["tss"], region["end"] - region["tss"]])
        plotdata_genes, plotdata_exons, plotdata_coding = get_genes_plotdata(
            region, genome=genome, window=window, use_cache=use_cache, only_canonical=only_canonical
        )

        if not show_others:
            plotdata_genes = plotdata_genes.query("ensembl_gene_id == @region.name").copy()
            plotdata_exons = plotdata_exons.query("gene == @region.name").copy()
            plotdata_coding = plotdata_coding.query("gene == @region.name").copy()

        return cls(
            plotdata_genes=plotdata_genes,
            plotdata_exons=plotdata_exons,
            plotdata_coding=plotdata_coding,
            region=region,
            gene_id=region.name,
            window=window,
            breaking = breaking,
            **kwargs,
        )