import itertools
import numpy as np
import pandas as pd


class NothingFilterer:
    def __init__(self):
        design = []
        for i in range(1):
            design.append({"i": 0})
        design = pd.DataFrame(design).set_index("i", drop=False)
        design.index.name = "i"
        design["ix"] = np.arange(len(design))
        self.design = design

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        yield data.coordinates[:, 0] > -999 * 10**10


class SizeFilterer:
    def __init__(self, window_size=100):
        cuts = np.arange(0, 800, step=window_size).tolist()

        design = []
        for window_start, window_end in zip(cuts[:-1], cuts[1:]):
            design.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_mid": window_start + (window_end - window_start) / 2,
                }
            )
        design = pd.DataFrame(design).set_index("window_mid", drop=False)
        design.index.name = "window"
        design["ix"] = np.arange(len(design))
        self.design = design

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        for design_ix, window_start, window_end in zip(
            self.design["ix"], self.design["window_start"], self.design["window_end"]
        ):
            sizes = data.coordinates[:, 1] - data.coordinates[:, 0]
            fragments_oi = ~((sizes > window_start) & (sizes < window_end))
            yield fragments_oi


def select_window(coordinates, window_start, window_end):
    """
    Selects coordinates of fragments that are within the window.
    """
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))


assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()


class WindowFilterer:
    def __init__(self, window, window_size=100):
        cuts = np.arange(*window, step=window_size).tolist() + [window[-1]]
        cuts = cuts

        design = []
        for window_start, window_end in zip(cuts[:-1], cuts[1:]):
            design.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_mid": window_start + (window_end - window_start) / 2,
                }
            )
        design = pd.DataFrame(design).set_index("window_mid", drop=False)
        design.index.name = "window"
        design["ix"] = np.arange(len(design))
        self.design = design

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        for design_ix, window_start, window_end in zip(
            self.design["ix"], self.design["window_start"], self.design["window_end"]
        ):
            fragments_oi = select_window(data.coordinates, window_start, window_end)
            yield fragments_oi


class ProvidedWindowFilterer(WindowFilterer):
    def __init__(self, design):
        design.index.name = "window"
        design["ix"] = np.arange(len(design))
        self.design = design


class WindowPairFilterer:
    def __init__(self, windows_oi):
        design = []
        for (window1_id, window1), (
            window2_id,
            window2,
        ) in itertools.combinations_with_replacement(windows_oi.iterrows(), 2):
            design.append(
                {
                    "window_start1": window1.window_start,
                    "window_end1": window1.window_end,
                    "window_mid1": int(window1.window_start + (window1.window_end - window1.window_start) // 2),
                    "window_start2": window2.window_start,
                    "window_end2": window2.window_end,
                    "window_mid2": int(window2.window_start + (window2.window_end - window2.window_start) // 2),
                    "window1": window1_id,
                    "window2": window2_id,
                }
            )
        design = pd.DataFrame(design)
        design.index = design["window_mid1"].astype(str) + "-" + design["window_mid2"].astype(str)
        design.index.name = "windowpair"
        design["ix"] = np.arange(len(design))
        self.design = design

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        for design_ix, window_start1, window_end1, window_start2, window_end2 in zip(
            self.design["ix"],
            self.design["window_start1"],
            self.design["window_end1"],
            self.design["window_start2"],
            self.design["window_end2"],
        ):
            fragments_oi = select_window(data.coordinates, window_start1, window_end1) & select_window(
                data.coordinates, window_start2, window_end2
            )
            yield fragments_oi


class WindowPairBaselineFilterer(WindowPairFilterer):
    def __init__(self, windowpair_filterer):
        self.design = windowpair_filterer.design.copy()

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        for design_ix, window_start1, window_end1, window_start2, window_end2 in zip(
            self.design["ix"],
            self.design["window_start1"],
            self.design["window_end1"],
            self.design["window_start2"],
            self.design["window_end2"],
        ):
            fragments_oi = ~(
                ~select_window(data.coordinates, window_start1, window_end1)
                & ~select_window(data.coordinates, window_start2, window_end2)
            )
            yield fragments_oi


class VariantFilterer(WindowFilterer):
    def __init__(self, positions, window_sizes=(1000,)):
        design = []
        for window_size in window_sizes:
            design.append(
                pd.DataFrame(
                    {
                        "window_start": positions - window_size // 2,
                        "window_end": positions + window_size // 2,
                        "window_mid": positions,
                        "window_size": window_size,
                    }
                ).reset_index()
            )
        design = pd.concat(design)
        design.index = pd.Series(
            design[positions.index.name] + "_" + design["window_size"].astype(str),
            name="variant_size",
        )
        design["ix"] = np.arange(len(design))
        self.design = design


def select_cutwindow(coordinates, window_start, window_end):
    # check whether coordinate 0 or coordinate 1 is within the window
    return ~(
        ((coordinates[:, 0] < window_end) & (coordinates[:, 0] > window_start))
        | ((coordinates[:, 1] < window_end) & (coordinates[:, 1] > window_start))
    )


class MultiWindowFilterer:
    def __init__(self, window, window_sizes=(100, 200), relative_stride=0.5):
        design = []
        for window_size in window_sizes:
            cuts = np.arange(*window, step=int(window_size * relative_stride))

            for window_start, window_end in zip(cuts, cuts + window_size):
                design.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_mid": window_start + (window_end - window_start) / 2,
                        "window_size": window_size,
                    }
                )
        design = pd.DataFrame(design)
        design.index = design["window_start"].astype(str) + "-" + design["window_end"].astype(str)
        design.index.name = "window"
        design["ix"] = np.arange(len(design))
        self.design = design

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        for design_ix, window_start, window_end in zip(
            self.design["ix"], self.design["window_start"], self.design["window_end"]
        ):
            fragments_oi = select_cutwindow(data.coordinates, window_start, window_end)
            yield fragments_oi


class WindowDirectionFilterer(WindowFilterer):
    def __init__(self, windows_oi, window, window_size=100):
        from chromatinhd.utils import crossing

        design = crossing(windows_oi, pd.DataFrame({"direction": ["forward", "reverse"]}))
        design.index = design["window_mid"].astype(str) + "_" + design["direction"]
        design.index.names = ["window_direction"]
        design["ix"] = np.arange(len(design))
        self.design = design

    def filter(self, data):
        for design_ix, window_start, window_end, direction in zip(
            self.design["ix"],
            self.design["window_start"],
            self.design["window_end"],
            self.design["direction"],
        ):
            if direction == "forward":
                fragments_oi = ~((data.coordinates[:, 0] < window_end) & (data.coordinates[:, 0] > window_start))
            else:
                fragments_oi = ~((data.coordinates[:, 1] < window_end) & (data.coordinates[:, 1] > window_start))
            yield fragments_oi


class WindowDirectionAll(WindowFilterer):
    def __init__(self, window, window_size=100):
        from chromatinhd.utils import crossing

        cuts = np.arange(*window, step=window_size).tolist() + [window[-1]]

        design_windows = []
        for window_start, window_end in zip(cuts[:-1], cuts[1:]):
            design_windows.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window": window_start + (window_end - window_start) / 2,
                }
            )

        cuts = np.arange(*window, step=window_size).tolist() + [window[-1]]

        design = crossing(
            pd.DataFrame(design_windows),
            pd.DataFrame({"direction": ["forward", "reverse"]}),
        )
        design.index = design["window"].astype(str) + "_" + design["direction"]
        design.index.names = ["window_direction"]
        design["ix"] = np.arange(len(design))
        self.design = design

    def filter(self, data):
        for design_ix, window_start, window_end, direction in zip(
            self.design["ix"],
            self.design["window_start"],
            self.design["window_end"],
            self.design["direction"],
        ):
            if direction == "forward":
                fragments_oi = ~((data.coordinates[:, 0] < window_end) & (data.coordinates[:, 0] > window_start))
            else:
                fragments_oi = ~((data.coordinates[:, 1] < window_end) & (data.coordinates[:, 1] > window_start))
            yield fragments_oi


class WindowSizeAll(WindowFilterer):
    def __init__(self, window, sizes, window_size=100):
        from chromatinhd.utils import crossing

        cuts = np.arange(*window, step=window_size).tolist() + [window[-1]]

        design_windows = []
        for window_start, window_end in zip(cuts[:-1], cuts[1:]):
            design_windows.append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window": window_start + (window_end - window_start) / 2,
                }
            )

        design_sizes = sizes.copy()
        design_sizes["size_start"] = design_sizes["start"]
        design_sizes["size_end"] = design_sizes["end"]
        design_sizes["size"] = design_sizes["mid"]

        design = crossing(
            pd.DataFrame(design_windows),
            pd.DataFrame(design_sizes),
        )
        design.index = design["window"].astype(str) + "_" + design["size"].astype(str)
        design.index.names = ["window_size"]
        design["ix"] = np.arange(len(design))
        self.design = design

    def filter(self, data):
        for design_ix, window_start, window_end, size_start, size_end in zip(
            self.design["ix"],
            self.design["window_start"],
            self.design["window_end"],
            self.design["size_start"],
            self.design["size_end"],
        ):
            sizes = data.coordinates[:, 1] - data.coordinates[:, 0]
            fragments_oi = ~(
                (data.coordinates[:, 0] < window_end)
                & (data.coordinates[:, 0] > window_start)
                & (sizes > size_start)
                & (sizes < size_end)
            )
            yield fragments_oi


class WindowSize(WindowFilterer):
    def __init__(self, windows, sizes):
        from chromatinhd.utils import crossing

        design_windows = windows.copy()
        design_windows["window"] = design_windows.index

        design_sizes = sizes.copy()
        design_sizes["size_start"] = design_sizes["start"]
        design_sizes["size_end"] = design_sizes["end"]
        design_sizes["size"] = design_sizes["mid"]

        design = crossing(
            pd.DataFrame(design_windows),
            pd.DataFrame(design_sizes),
        )
        design.index = design["window"].astype(str) + "_" + design["size"].astype(str)
        design.index.names = ["window_size"]
        design["ix"] = np.arange(len(design))
        self.design = design

    def filter(self, data):
        for design_ix, window_start, window_end, size_start, size_end in zip(
            self.design["ix"],
            self.design["window_start"],
            self.design["window_end"],
            self.design["size_start"],
            self.design["size_end"],
        ):
            sizes = data.coordinates[:, 1] - data.coordinates[:, 0]
            fragments_oi = select_window(data.coordinates, window_start, window_end)
            fragments_oi = ~((~fragments_oi) & (sizes > size_start) & (sizes <= size_end))
            yield fragments_oi


class SizeIntervalFilterer(SizeFilterer):
    def __init__(self, start, end, window_size=100):
        super().__init__(window_size=window_size)
        self.start = start
        self.end = end

    def setup_next_chunk(self):
        return len(self.design)

    def filter(self, data):
        for design_ix, window_start, window_end in zip(
            self.design["ix"], self.design["window_start"], self.design["window_end"]
        ):
            sizes = data.coordinates[:, 1] - data.coordinates[:, 0]
            fragments_oi = ~select_window(data.coordinates, self.start, self.end)
            fragments_oi = ~((sizes > window_start) & (sizes < window_end) & fragments_oi)
            yield fragments_oi
