from chromatinhd.grid.grid import Grid, Panel
import numpy as np
import pandas as pd
import matplotlib as mpl


class Broken(Grid):
    def __init__(self, regions, width, gap=1, height=0.5, *args, **kwargs):
        super().__init__(padding_width=gap, *args, **kwargs)

        regions["width"] = regions["end"] - regions["start"]
        regions["ix"] = np.arange(len(regions))

        assert width > gap * (len(regions) - 1), "Width is not enough to contain gaps"

        self.resolution = regions["width"].sum() / (width - (len(regions) - 1) * gap)

        for region, region_info in regions.iterrows():
            subpanel_width = region_info["width"] / self.resolution
            panel, ax = self.add_right(
                Panel((subpanel_width, height)),
            )

            ax.set_xlim(region_info["start"], region_info["end"])
            ax.set_xticks([])
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            if region_info["ix"] != 0:
                ax.spines.left.set_visible(False)
            if region_info["ix"] != len(regions) - 1:
                ax.spines.right.set_visible(False)
            ax.spines.top.set_visible(False)


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
