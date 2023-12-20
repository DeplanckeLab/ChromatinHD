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
    def __init__(self, window, window_size=100):
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

    def __call__(self, coordinates):
        precomputed = select_cutwindow_multiple(
            coordinates, self.design["window_start"].values, self.design["window_end"].values
        )
        for i, (window_start, window_end) in enumerate(zip(self.design["window_start"], self.design["window_end"])):
            if np.isnan(window_start):
                fragments_oi = None
            else:
                fragments_oi = precomputed[i]
            yield fragments_oi
