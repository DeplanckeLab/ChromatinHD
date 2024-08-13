from chromatinhd.grid.grid import Grid, Panel
import numpy as np
import pandas as pd
import dataclasses


@dataclasses.dataclass
class Breaking:
    regions: pd.DataFrame
    gap: int = 0.05
    resolution: int = 2500

    @property
    def width(self):
        return (self.regions["length"] / self.resolution).sum() + self.gap * (len(self.regions) - 1)


class Broken(Grid):
    """
    A grid build from distinct regions that are using the same coordinate space
    """

    def __init__(self, breaking, height=0.5, margin_height=0.0, *args, **kwargs):
        super().__init__(padding_width=breaking.gap, margin_height=margin_height, *args, **kwargs)

        regions = breaking.regions

        regions["width"] = regions["end"] - regions["start"]
        regions["ix"] = np.arange(len(regions))

        for i, (region, region_info) in enumerate(regions.iterrows()):
            if "resolution" in region_info.index:
                resolution = region_info["resolution"]
            else:
                resolution = breaking.resolution
            subpanel_width = region_info["width"] / resolution
            panel, ax = self.add_right(
                Panel((subpanel_width, height + 1e-4)),
            )

            ax.set_xlim(region_info["start"], region_info["end"])
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            if i != 0:
                ax.set_yticks([])
            if region_info["ix"] != 0:
                ax.spines.left.set_visible(False)
            if region_info["ix"] != len(regions) - 1:
                ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.set_facecolor("none")

            # ax.plot([0, 0], [0, 1], transform=ax.transAxes, color="k", lw=1, clip_on=False)


class BrokenGrid(Grid):
    """
    A grid build from distinct regions that are using the same coordinate space
    """

    def __init__(self, breaking, height=0.5, padding_height=0.05, margin_height=0.0, *args, **kwargs):
        super().__init__(padding_width=breaking.gap, margin_height=margin_height, *args, **kwargs)

        regions = breaking.regions

        regions["width"] = regions["end"] - regions["start"]
        regions["ix"] = np.arange(len(regions))

        regions["panel_width"] = regions["width"] / breaking.resolution

        self.panel_widths = regions["panel_width"].values

        for i, (region, region_info) in enumerate(regions.iterrows()):
            _ = self.add_right(
                Grid(padding_height=padding_height, margin_height=0.0),
            )


def add_slanted_x(ax1, ax2, size=4, **kwargs):
    d = 1.0  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=size,
        linestyle="none",
        mew=1,
        clip_on=False,
        **{"color": "k", "mec": "k", **kwargs},
    )
    ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)


class TransformBroken:
    def __init__(self, breaking, width=None):
        """
        Transforms from data coordinates to (broken) data coordinates

        Parameters
        ----------
        breaking : pd.DataFrame
            Regions to break
        resolution : float
            Resolution of the data to go from data to points
        gap : float
            Gap between the regions in points

        """

        regions = breaking.regions

        regions["width"] = regions["end"] - regions["start"]
        regions["ix"] = np.arange(len(regions))

        regions["cumstart"] = (np.pad(np.cumsum(regions["width"])[:-1], (1, 0))) + regions[
            "ix"
        ] * breaking.gap * breaking.resolution
        regions["cumend"] = np.cumsum(regions["width"]) + regions["ix"] * breaking.gap / breaking.resolution

        self.regions = regions
        self.resolution = breaking.resolution
        self.gap = breaking.gap

    def __call__(self, x):
        """
        Transform from data coordinates to (broken) data coordinates

        Parameters
        ----------
        x : float
            Position in data coordinates

        Returns
        -------
        float
            Position in (broken) data coordinates

        """

        assert isinstance(x, (int, float, np.ndarray, np.float64, np.int64))

        if isinstance(x, (int, float, np.float64, np.int64)):
            x = np.array([x])

        match = (x[:, None] >= self.regions["start"].values) & (x[:, None] <= self.regions["end"].values)

        argmax = np.argmax(
            match,
            axis=1,
        )
        allzero = (match == False).all(axis=1)

        # argmax[allzero] = np.nan

        y = self.regions.iloc[argmax]["cumstart"].values + (x - self.regions.iloc[argmax]["start"].values)
        y[allzero] = np.nan

        return y
