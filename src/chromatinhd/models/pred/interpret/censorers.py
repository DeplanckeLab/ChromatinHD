import pandas as pd
import numpy as np
import torch


def select_cutwindow(coordinates, window_start, window_end):
    """
    check whether coordinate 0 or coordinate 1 is within the window
    """
    return ~(
        ((coordinates[:, 0] < window_end) & (coordinates[:, 0] > window_start))
        | ((coordinates[:, 1] < window_end) & (coordinates[:, 1] > window_start))
    )


def select_cutwindow_multiple(coordinates, window_start, window_end):
    """
    check whether coordinate 0 or coordinate 1 is within the window
    """
    window_start = torch.from_numpy(window_start).to(coordinates.device)
    window_end = torch.from_numpy(window_end).to(coordinates.device)
    return ~(
        ((coordinates[:, 0][None, :] < window_end[:, None]) & (coordinates[:, 0][None, :] > window_start[:, None]))
        | ((coordinates[:, 1][None, :] < window_end[:, None]) & (coordinates[:, 1][None, :] > window_start[:, None]))
    )


class WindowCensorer:
    def __init__(self, window, window_size=200):
        design = [{"window": "control"}]
        cuts = np.arange(*window, step=int(window_size))

        for window_start, window_end in zip(cuts, cuts + window_size):
            if window_start < window[0]:
                continue
            if window_end > window[1]:
                continue
            design.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_mid": window_start + (window_end - window_start) / 2,
                    "window": f"{window_start}-{window_end}",
                }
            )
        design = pd.DataFrame(design).set_index("window")
        assert design.index.is_unique
        design["ix"] = np.arange(len(design))
        self.design = design

    def __len__(self):
        return len(self.design)

    def __call__(self, data):
        for window_start, window_end in zip(self.design["window_start"], self.design["window_end"]):
            if np.isnan(window_start):
                fragments_oi = None
            else:
                fragments_oi = select_cutwindow(data.fragments.coordinates, window_start, window_end)
            yield fragments_oi


class MultiWindowCensorer:
    def __init__(self, window, window_sizes=(50, 100, 200, 500), relative_stride=0.5):
        design = [{"window": "control"}]
        for window_size in window_sizes:
            cuts = np.arange(*window, step=int(window_size * relative_stride))

            for window_start, window_end in zip(cuts, cuts + window_size):
                if window_start < window[0]:
                    continue
                if window_end > window[1]:
                    continue
                design.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_mid": window_start + (window_end - window_start) / 2,
                        "window_size": window_size,
                        "window": f"{window_start}-{window_end}",
                    }
                )
        design = pd.DataFrame(design).set_index("window")
        assert design.index.is_unique
        design["ix"] = np.arange(len(design))
        self.design = design

    def __len__(self):
        return len(self.design)

    def __call__(self, data):
        precomputed = select_cutwindow_multiple(
            data.fragments.coordinates, self.design["window_start"].values, self.design["window_end"].values
        )
        for i, (window_start, window_end) in enumerate(zip(self.design["window_start"], self.design["window_end"])):
            if np.isnan(window_start):
                fragments_oi = None
            else:
                fragments_oi = precomputed[i]
            yield fragments_oi


class SizeCensorer:
    def __init__(self, width):
        design = [{"size": "control"}]
        for size in range(0, 800, width):
            design.append(
                {
                    "start": size,
                    "end": size + width,
                    "window": size + width // 2,
                }
            )
        design = pd.DataFrame(design).set_index("window")
        assert design.index.is_unique
        design["ix"] = np.arange(len(design))
        self.design = design

    def __len__(self):
        return len(self.design)

    def __call__(self, data):
        fragment_size = (data.fragments.coordinates[:, 0] - data.fragments.coordinates[:, 1]).abs()
        for i, (window_start, window_end) in enumerate(zip(self.design["start"], self.design["end"])):
            if np.isnan(window_start):
                fragments_oi = None
            else:
                fragments_oi = ~((fragment_size > window_start) & (fragment_size <= window_end))
            yield fragments_oi


class WindowSizeCensorer:
    def __init__(self, windows, sizes):
        from chromatinhd.utils import crossing

        design_windows = windows.copy()
        design_windows["window"] = design_windows.index
        self.design_windows = design_windows

        design_sizes = sizes.copy()
        design_sizes["size_start"] = design_sizes["start"]
        design_sizes["size_end"] = design_sizes["end"]
        design_sizes["size"] = design_sizes["mid"]

        self.design_sizes = design_sizes

        design = crossing(
            pd.DataFrame(design_windows),
            pd.DataFrame(design_sizes),
        )
        design.index = design["window"].astype(str) + "_" + design["size"].astype(str)
        design.index.names = ["window_size"]

        design = pd.concat(
            [
                pd.DataFrame({"window": ["control"], "size": ["control"]}, index=["control_control"]),
                design,
            ]
        )

        design["ix"] = np.arange(len(design))
        self.design = design

    def __len__(self):
        return len(self.design)

    def __call__(self, data):
        precomputed = select_cutwindow_multiple(
            data.fragments.coordinates,
            self.design_windows["window_start"].values,
            self.design_windows["window_end"].values,
        )
        precomputed = {window: precomputed[i] for i, window in enumerate(self.design_windows.index)}

        fragment_sizes = (data.fragments.coordinates[:, 0] - data.fragments.coordinates[:, 1]).abs()
        precomputed_sizes = {}
        for size, size_start, size_end in zip(
            self.design_sizes["mid"], self.design_sizes["start"], self.design_sizes["end"]
        ):
            precomputed_sizes[size] = ~((fragment_sizes > size_start) & (fragment_sizes <= size_end))

        for window, size in zip(self.design["window"], self.design["size"]):
            if window == "control":
                fragments_oi = None
            else:
                fragments_oi = precomputed[window] | precomputed_sizes[size]

            yield fragments_oi
