import polyptich as pp
import pandas as pd
import numpy as np

import matplotlib as mpl

class TracksBroken(pp.grid.broken.Broken):
    """
    Visualize multiple tracks in a broken axis plot.

    Parameters
    ----------
    design:
        Design of the tracks to plot. Has to have the following columns:
        - `bw`: bigwig file path
        - `name`: name of the track
        optionally:
        - `color`: color of the track
    region:
        Dictionary/pd.Series with the following
        - `chrom`: chromosome
        - `start`: start position
        - `end`: end position
        - `strand`: strand
    breaking : pp.grid.Breaking
        Breaking object
    window:
        Window size to plot. If None, the whole region is plotted.
    height:
        Height of the plot. Default is 0.6.
    scale_per_region:
        If True, the scale is set to the maximum value of each region. Default is False, which uses a common scale.

    """

    def __init__(
        self, design, region, breaking, window=None, height=0.6, scale_per_region = False, **kwargs
    ):
        super().__init__(
            breaking=breaking,
            height=height,
            **kwargs,
        )

        if "tss" in region:
            pad = region["start"] - region["tss"]
        else:
            pad = 0

        plotdata = pd.DataFrame(
            {
                "pos": np.arange(region["end"] - region["start"]) + pad,
            }
        )
        for k, bw in design["bw"].items():
            plotdata[k] = bw.values(region["chrom"], region["start"], region["end"])[
                :: region["strand"]
            ]

        for i, (region, region_info), (panel, ax) in zip(
            range(len(breaking.regions)), breaking.regions.iterrows(), self.elements[0]
        ):
            plotdata_region = plotdata.loc[
                (plotdata["pos"] >= region_info["start"])
                & (plotdata["pos"] <= region_info["end"])
            ]
            if len(plotdata_region) > 0:
                for k in design.index:
                    ax.fill_between(
                        plotdata_region["pos"],
                        0,
                        plotdata_region[k],
                        label=k,
                        lw=0.,
                        color="#33333366",
                    )
            ax.set_ylim(0, 50)
            ax.set_yticks([], minor = True)
            ax.set_yticks([])
            ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            # top ytick vertical alignment top
            if not scale_per_region and (i == 0):
                ax.set_yscale("symlog", linthresh=40)
                ax.set_yticks([0, 10, 100])
                ax.set_yticks([0, 2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], minor = True)
                ax.get_yticklabels()[0].set_verticalalignment("bottom")
                ax.get_yticklabels()[-1].set_verticalalignment("top")
                ax.tick_params(axis="y", length=4, labelsize=6, pad = 1)
            elif not scale_per_region:
                pass
            else:
                ax.set_yscale("linear")
                ax.set_yticks([0, plotdata_region[k].max().round(0)-1])
                ax.annotate(
                    str(plotdata_region[k].max().round(0)),
                    (0, 1),
                    xycoords="axes fraction",
                    xytext=(2, -5),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=6,
                )
                ax.set_ylim(0, plotdata_region[k].max())
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=-4, labelsize=6, pad = 1)
                # ax.set_yscale("symlog", linthresh=40)
    @classmethod
    def from_bigwigs(
        cls,
        design: pd.DataFrame,
        region: pd.Series,
        breaking: pp.grid.Breaking,
        window=None,
        height=0.6,
        **kwargs,
    ):
        """
        Visualize multiple tracks in a broken axis plot.

        Parameters
        ----------
        design:
            Design of the tracks to plot. Has to have the following columns:
            - `bw`: bigwig file path
            - `name`: name of the track
            optionally:
            - `color`: color of the track
        region:
            Dictionary/pd.Series with the following
            - `chrom`: chromosome
            - `start`: start position
            - `end`: end position
            - `strand`: strand
        breaking : pp.grid.Breaking
            Breaking object
        """
        try:
            import pyBigWig
        except ImportError:
            raise ImportError(
                "pyBigWig is required for this method. Please install it using `pip install pyBigWig`."
            )
        design["bw"] = [pyBigWig.open(f"{sample}.bw") for sample in design["sample"]]
        return TracksBroken(
            design=design,
            region=region,
            breaking=breaking,
            window=window,
            height=height,
            **kwargs,
        )


class Tracks(pp.grid.Panel):
    """
    Visualize multiple tracks from BigWig or BAM files in an axis plot.

    Parameters
    ----------
    design : pd.DataFrame
        Design of the tracks to plot. Must have a 'name' column.
        For each track, provide EITHER:
        - `bw`: path to the bigwig file
        OR
        - `bam`: path to the bam file
        Optionally:
        - `color`: color string for the track (e.g., '#FF0000', 'blue')
    region : pd.Series or dict
        Dictionary/pd.Series with the following:
        - `chrom`: chromosome name
        - `start`: start position (0-based)
        - `end`: end position
        - `strand`: strand ('+' or '-' or 1 or -1). Determines data orientation if applicable.
    width : float
        Width of the panel.
    window : int, optional
        Window size to plot. If None, the whole region is plotted.
        (Currently not implemented in data fetching, visual only if polyptich uses it)
    height : float, default 0.6
        Height of the plot.
    scale_per_region : bool, default False
        If True, the Y-axis scale is set to the maximum value within this panel.
        If False, a common scale (symlog, 0-100 with specific ticks) is used,
        unless overridden by set_ylim externally.
    default_track_color : str, default "#33333366"
        Default color for tracks if not specified in the design.
    **kwargs :
        Additional keyword arguments passed to pp.grid.Panel.
    """

    def __init__(
        self,
        design: pd.DataFrame,
        region: pd.Series,
        width: float,
        window=None, # Note: window is not used in data processing here
        height=0.6,
        bw=None,
        scale_per_region=False,
        default_track_color="#33333366",
        **kwargs,
    ):
        super().__init__(
            (width, height),
            **kwargs,
        )

        if not isinstance(design, pd.DataFrame):
            raise ValueError("design must be a pandas DataFrame.")
        if not all(col in design.columns for col in ['name']):
            raise ValueError("design DataFrame must have a 'name' column.")
        if not all(col in region for col in ['chrom', 'start', 'end']):
            raise ValueError("region must contain 'chrom', 'start', and 'end'.")

        # Standardize strand: 1 for positive, -1 for negative
        raw_strand = region.get("strand", "+")
        if isinstance(raw_strand, str):
            strand_modifier = -1 if raw_strand == "-" else 1
        elif isinstance(raw_strand, (int, float)):
            strand_modifier = -1 if raw_strand < 0 else 1
        else:
            strand_modifier = 1 # Default to positive strand

        region_len = region["end"] - region["start"]
        if region_len <= 0:
            # print(f"Warning: Region length is non-positive for {region['chrom']}:{region['start']}-{region['end']}. Skipping plot.")
            self.set_xticks([])
            self.set_yticks([])
            return

        if "tss" in region:
            # If TSS is present, positions are relative to TSS
            # If region start is 500, TSS is 600, pad is -100. First pos is -100.
            # If region start is 600, TSS is 500, pad is 100. First pos is 100.
            pad = region["start"] - region["tss"]
        else:
            pad = 0 # Positions will be 0 to region_len-1

        # X-axis positions
        # If strand_modifier is -1, we want positions to run from high to low effectively,
        # but the data array itself will be reversed. The 'pos' array should map to these data points.
        # Example: region 100-200, TSS 150.
        # pad = 100 - 150 = -50
        # pos = [0..99] - 50 = [-50 .. 49]
        # If strand is '-', data is reversed. pos still maps to the reversed data.
        plotdata = pd.DataFrame(
            {"pos": np.arange(region_len) + pad}
        )

        all_values_for_panel_scaling = []

        for idx, track_info in design.iterrows():
            track_name = track_info["name"]
            values = np.zeros(region_len, dtype=float) # Default to zeros

            if "bw" in track_info and pd.notna(track_info["bw"]):
                try:
                    bw_path = track_info["bw"]
                    import pyBigWig
                    bw_file = pyBigWig.open(str(bw_path))
                    # pyBigWig returns None if chromosome not found or no data
                    raw_values = bw_file.values(region["chrom"], region["start"], region["end"])
                    if raw_values is not None:
                        values = np.nan_to_num(raw_values, nan=0.0)
                    else:
                        print(f"Warning: No data for {track_name} in {bw_path} for {region['chrom']}:{region['start']}-{region['end']}.")
                    bw_file.close()
                except Exception as e:
                    print(f"Error processing BigWig {track_info['bw']} for track {track_name}: {e}")
            
            elif "bam" in track_info and pd.notna(track_info["bam"]):
                try:
                    import pysam
                    bam_path = track_info["bam"]
                    bam_file = pysam.AlignmentFile(str(bam_path), "rb")
                    # count_coverage returns (counts_A, counts_C, counts_G, counts_T)
                    # We need to ensure the length matches region_len.
                    # pysam's count_coverage is 0-based for start and end.
                    coverage_arrays = bam_file.count_coverage(
                        contig=region["chrom"], start=region["start"], stop=region["end"]
                    )
                    # Summing counts for A, C, G, T to get total coverage per position
                    current_values = np.sum(coverage_arrays, axis=0)
                    
                    # Ensure the length matches region_len, pad with zeros if necessary
                    # This can happen if region extends beyond chromosome bounds in BAM
                    if len(current_values) < region_len:
                        padded_values = np.zeros(region_len, dtype=float)
                        padded_values[:len(current_values)] = current_values
                        values = padded_values
                    else:
                        values = current_values[:region_len] # Truncate if longer (shouldn't happen with correct pysam use)
                        
                    bam_file.close()
                except ValueError as e: # Often for "fetch reference sequence" if contig not in BAM
                     print(f"Warning: Could not fetch {region['chrom']} for {track_name} in {bam_path}. {e}")
                except Exception as e:
                    print(f"Error processing BAM {track_info['bam']} for track {track_name}: {e}")
            else:
                # print(f"Warning: No 'bw' or 'bam' path specified for track {track_name}.")
                # No data to plot for this track, 'values' remains zeros
                pass

            if bw is not None:
                # smooth by bandwidth
                if not isinstance(bw, (int, float)):
                    raise ValueError("bw must be a numeric value (int or float).")
                if bw > 0:
                    # Apply a simple moving average smoothing
                    values = np.convolve(values, np.ones(int(bw)) / bw, mode='same')

            # Apply strand orientation to the data values
            plotdata[track_name] = values[::strand_modifier]
            all_values_for_panel_scaling.extend(plotdata[track_name].tolist())

        if not all_values_for_panel_scaling: # No data was successfully loaded
            max_val_this_panel = 50 # Default if no data
        else:
            max_val_this_panel = np.max(all_values_for_panel_scaling)
            if max_val_this_panel == 0: # All data is zero
                 max_val_this_panel = 1.0 # Avoid division by zero or empty range for y-axis

        # Plotting
        if len(plotdata.columns) > 1: # At least one track data column besides 'pos'
            for idx, track_info in design.iterrows():
                track_name = track_info["name"]
                track_color = track_info.get("color", default_track_color)
                if track_name in plotdata: # Ensure data was actually loaded for this track
                    self.fill_between(
                        plotdata["pos"],
                        0,
                        plotdata[track_name],
                        label=track_name, # Polyptich might not use matplotlib labels directly
                        lw=0.0,
                        color=track_color,
                        zorder=idx + 1 # Simple z-ordering
                    )
                    # Polyptich typically handles legends differently, but you can add artists
                    # self.text(1.02, idx * 0.1, track_name, transform=self.transAxes, color=track_color, fontsize=6)


        # Y-axis configuration
        self.set_yticks([], minor=True)
        self.set_yticks([])
        self.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

        if scale_per_region:
            self.set_yscale("linear")
            # Ensure ylim_max is slightly larger than max_val_this_panel if max_val_this_panel is not 0
            ylim_max = max_val_this_panel * 1.05 if max_val_this_panel > 0 else 1.0
            self.set_ylim(0, ylim_max)
            
            # Determine a sensible tick for the max value
            # Avoid tick at 0 if max_val_this_panel is very small.
            top_tick_val = float(f"{max_val_this_panel:.1g}") # Use scientific notation for general applicability
            if top_tick_val == 0 and max_val_this_panel > 0: # e.g. max_val = 0.004, .1g -> 0
                 top_tick_val = max_val_this_panel

            if max_val_this_panel > 0: # Only add ticks if there's data
                self.set_yticks([0, top_tick_val])
                self.set_yticklabels(["0", f"{top_tick_val:.1g}"]) # Format it similarly
                self.get_yticklabels()[0].set_verticalalignment("bottom")
                if len(self.get_yticklabels()) > 1:
                    self.get_yticklabels()[-1].set_verticalalignment("top")

            # Annotate max value (original style)
            # self.annotate(
            #     f"{max_val_this_panel:.2f}", # Or use .0f for integer-like
            #     (0, 1),
            #     xycoords="axes fraction",
            #     xytext=(2, -2), # Adjusted offset
            #     textcoords="offset points",
            #     ha="left",
            #     va="top", # Changed to top for consistency with tick label
            #     fontsize=6,
            # )
            self.tick_params(axis="y", length=4, labelsize=6, pad=1) # Changed length to be positive
        else:
            # Common scale logic (original symlog approach)
            # This part assumes a general range, e.g., for ChIP-seq like data.
            # It might need adjustment based on typical data values.
            self.set_ylim(0, max(100, max_val_this_panel * 1.05)) # Ensure ylim covers at least 0-100 or actual max
            self.set_yscale("symlog", linthresh=40) # linthresh=40 means linear behavior from -40 to 40
            
            # Sensible ticks for symlog with linthresh=40
            # Adapt ticks based on actual max_val_this_panel
            symlog_ticks = [0, 10]
            if max_val_this_panel > 40: # only add 100 if data goes significantly beyond linthresh
                 symlog_ticks.append(100)
            if max_val_this_panel > 1000: # example for larger values
                 symlog_ticks.append(1000)

            # Filter ticks to be within or near actual data range
            current_ylim = self.get_ylim()
            symlog_ticks = [t for t in symlog_ticks if t <= current_ylim[1] * 1.1]
            if not symlog_ticks or symlog_ticks[-1] < max_val_this_panel * 0.8 : # if highest tick is too small
                if max_val_this_panel > 0:
                    # add a tick close to the max value
                    if max_val_this_panel not in symlog_ticks:
                        symlog_ticks.append(float(f"{max_val_this_panel:.1g}"))
                        symlog_ticks = sorted(list(set(symlog_ticks))) # remove duplicates and sort

            self.set_yticks(symlog_ticks)

            # Minor ticks (example)
            # self.set_yticks([0, 2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], minor=True)
            
            if self.get_yticklabels():
                self.get_yticklabels()[0].set_verticalalignment("bottom")
                if len(self.get_yticklabels()) > 1:
                    self.get_yticklabels()[-1].set_verticalalignment("top")
            self.tick_params(axis="y", length=4, labelsize=6, pad=1)
            
        # X-axis (common setup)
        window = [
            region["start"] - region.get("tss", 0),  # Start position, or TSS if available
            region["end"] - region.get("tss", 0)     # End position, or TSS if available
        ]
        if window is not None:
            self.set_xlim(window[0], window[1])
        else:
            if region["strand"] == 1:
                self.set_xlim(region["start"], region["end"])
            else:
                self.set_xlim(region["end"], region["start"])
        self.spines['top'].set_visible(False)
        self.spines['right'].set_visible(False)

        self.set_xticks([]) # If you don't want x-ticks
        # self.set_xticks([]) # If you don't want x-ticks


    @classmethod
    def from_files( # Renamed for clarity, can be adapted
        cls,
        design: pd.DataFrame,
        region: pd.Series,
        width: float,
        **kwargs, # To pass height, scale_per_region etc.
    ):
        """
        Convenience classmethod to create a Tracks panel.
        This is now mostly a direct pass-through to __init__ as __init__ handles paths.

        Parameters are the same as __init__.
        """
        # If breaking object is a core part of polyptich's panel, pass it:
        # return cls(design=design, region=region, width=width, breaking=breaking, **kwargs)
        return cls(design=design, region=region, width=width, **kwargs)
    
    @classmethod
    def from_file(
        cls,
        file,
        region: pd.Series,
        width: float,
        **kwargs,
    ):
        """
        Convenience classmethod to create a Tracks panel with breaking.
        
        Parameters are the same as __init__.
        """
        import pathlib
        if isinstance(file, pathlib.Path):
            file = str(file)
        if not (file.endswith(".bw") or file.endswith(".bam")):
            raise ValueError("file must be a path to a .bw or .bam file.")
        
        design = pd.DataFrame({
            "name": [file.split("/")[-1].replace(".bw", "").replace(".bam", "")],
            "bw": [file] if file.endswith(".bw") else None,
            "bam": [file] if file.endswith(".bam") else None,
            "color": ["#33333366"],  # Default color
        })
        return cls(design=design, region=region, width=width, **kwargs)