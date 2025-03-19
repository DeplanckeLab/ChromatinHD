from matplotlib.path import Path
import matplotlib as mpl
import numpy as np
import copy
import pandas as pd

from polyptich.grid import Grid, Panel, Broken

path_a = Path(np.array([[0.65971097, 0.78427118],
       [0.33386837, 0.78427118],
       [0.27528096, 1.        ],
       [0.        , 1.        ],
       [0.34028903, 0.        ],
       [0.65971097, 0.        ],
       [1.        , 1.        ],
       [0.71829837, 1.        ],
       [0.65971097, 0.78427118],
       [0.65971097, 0.78427118],
       [0.37560198, 0.61255405],
       [0.61637226, 0.61255405],
       [0.49598698, 0.17027416],
       [0.37560198, 0.61255405],
       [0.37560198, 0.61255405]]), np.array([ 1,  2,  2,  2,  2,  2,  2,  2,  2, 79,  1,  2,  2,  2, 79],
      dtype=np.uint8))


path_t = Path(np.array([[0.6588729 , 0.18109665],
       [0.6688729 , 1.        ],
       [0.34746375, 1.        ],
       [0.34746375, 0.18109665],
       [0.        , 0.18109665],
       [0.        , 0.        ],
       [1.        , 0.        ],
       [1.        , 0.18109665],
       [0.6688729 , 0.18109665],
       [0.6688729 , 0.18109665]]), np.array([ 1,  2,  2,  2,  2,  2,  2,  2,  2, 79], dtype=np.uint8))


path_c = Path(np.array([[5.83258302e-01, 0.00000000e+00],
       [7.17371840e-01, 0.00000000e+00],
       [8.10981077e-01, 2.74914295e-02],
       [9.05490696e-01, 5.42953515e-02],
       [9.83798468e-01, 1.03780069e-01],
       [8.38884117e-01, 2.37113370e-01],
       [7.87578689e-01, 2.04810826e-01],
       [7.25472763e-01, 1.86254117e-01],
       [6.63366523e-01, 1.67010620e-01],
       [5.90459157e-01, 1.67010620e-01],
       [5.12151071e-01, 1.67010620e-01],
       [4.47344628e-01, 2.01374967e-01],
       [3.82538185e-01, 2.35051807e-01],
       [3.43834332e-01, 3.08591375e-01],
       [3.05130480e-01, 3.81443675e-01],
       [3.05130480e-01, 4.98282071e-01],
       [3.05130480e-01, 6.70103325e-01],
       [3.86138455e-01, 7.48453768e-01],
       [4.68046812e-01, 8.26116942e-01],
       [5.95859563e-01, 8.26116942e-01],
       [6.89469114e-01, 8.26116942e-01],
       [7.51575354e-01, 7.99312780e-01],
       [8.13681594e-01, 7.72508858e-01],
       [8.65886776e-01, 7.40893582e-01],
       [1.00000000e+00, 8.71477813e-01],
       [9.29793151e-01, 9.24398629e-01],
       [8.27182923e-01, 9.62199314e-01],
       [7.24572381e-01, 1.00000000e+00],
       [5.79658031e-01, 1.00000000e+00],
       [4.10440911e-01, 1.00000000e+00],
       [2.79027889e-01, 9.42955338e-01],
       [1.48514621e-01, 8.85223408e-01],
       [7.38071197e-02, 7.73883394e-01],
       [0.00000000e+00, 6.61855872e-01],
       [0.00000000e+00, 4.98282071e-01],
       [0.00000000e+00, 3.38831875e-01],
       [7.65076369e-02, 2.27491861e-01],
       [1.53915341e-01, 1.16151608e-01],
       [2.86228431e-01, 5.84196776e-02],
       [4.18541834e-01, 4.79768063e-07],
       [5.83258302e-01, 4.79768063e-07],
       [5.83258302e-01, 0.00000000e+00],
       [5.83258302e-01, 0.00000000e+00]]), np.array([ 1,  3,  3,  3,  3,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  3,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,
        3,  3,  3,  3,  3,  3,  3,  2, 79], dtype=np.uint8))


path_g = Path(np.array([[0.55545272, 1.        ],
       [0.28505905, 1.        ],
       [0.14207124, 0.87216502],
       [0.        , 0.74364277],
       [0.        , 0.49828183],
       [0.        , 0.33608249],
       [0.07882659, 0.22542945],
       [0.15765318, 0.11408938],
       [0.29147501, 0.05704469],
       [0.42529716, 0.        ],
       [0.59028319, 0.        ],
       [0.72960473, 0.        ],
       [0.82492983, 0.03230232],
       [0.92025524, 0.06460464],
       [0.994499  , 0.11752572],
       [0.8368455 , 0.23986256],
       [0.77910057, 0.20137458],
       [0.72593846, 0.18281787],
       [0.67277604, 0.16426043],
       [0.60494856, 0.16426043],
       [0.51970569, 0.16426043],
       [0.45279477, 0.19862455],
       [0.38680043, 0.23230165],
       [0.3483037 , 0.30652852],
       [0.30980697, 0.38006788],
       [0.30980697, 0.49965565],
       [0.30980697, 0.62405429],
       [0.33638898, 0.69759389],
       [0.36388596, 0.77113325],
       [0.41888183, 0.80343581],
       [0.47387739, 0.83505086],
       [0.55820338, 0.83505086],
       [0.60311639, 0.83505086],
       [0.64252968, 0.82748972],
       [0.68194298, 0.8192425 ],
       [0.71585688, 0.80618321],
       [0.71585688, 0.58831611],
       [0.55637025, 0.58831611],
       [0.52704014, 0.43024061],
       [1.        , 0.43024061],
       [1.        , 0.90378007],
       [0.9046749 , 0.94776643],
       [0.79468411, 0.97388333],
       [0.68560988, 1.        ],
       [0.55545432, 1.        ],
       [0.55545272, 1.        ],
       [0.55545272, 1.        ]]), np.array([ 1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  3,
        3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
        3,  3,  2,  2,  2,  2,  2,  3,  3,  3,  3,  2, 79], dtype=np.uint8))


# path_c = path_g = path_a = path_t = Path(np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]), np.array([1, 2, 2, 2, 79], dtype=np.uint8))
polygon_c = mpl.patches.PathPatch(
    path_c
)
polygon_a = mpl.patches.PathPatch(
    path_a
)
polygon_t = mpl.patches.PathPatch(
    path_t
)
polygon_g = mpl.patches.PathPatch(
    path_g
)

polygons = {
    "A": polygon_a,
    "T": polygon_t,
    "G": polygon_g,
    "C": polygon_c,
}
polygon_colors = {
    "A": "#2ECC40",
    "T": "#FF4136",
    "G": "#FFDC00",
    "C": "#0074D9",
}



def plot_sequence(sequence, x):
    patches = []
    colors = []
    import copy
    for i, char in enumerate(sequence):
        polygon = copy.copy(polygons[char.upper()])
        polygon.set_transform(mpl.transforms.Affine2D().translate(x + i, -1).scale(1, -1))

        patches.append(polygon)
        colors.append(polygon_colors[char.upper()])
    collection = mpl.collections.PatchCollection(
        patches,
        lw=0,
        facecolor=colors,
    )
    return collection

def plot_motif(pwm, x, y):
    """
    Plot a motif at a given x and y position.
    """
    patches = []
    colors = []
    characters = np.array(["A", "C", "G", "T"])
    for row in range(pwm.shape[0]):
        pos = 0
        pwm_position = pwm[row] / np.sqrt(2)
        order = np.argsort(pwm_position)[::-1]
        for score, char in zip(pwm_position[order], characters[order]):
            if score > 0:
                patch = copy.copy(polygons[char.upper()])
                patches.append(patch)
                patch.set_transform(mpl.transforms.Affine2D().scale(1, -score).translate(x+row, y+pos+score))
                pos += score
                colors.append(polygon_colors[char.upper()])

    collection = mpl.collections.PatchCollection(
        patches,
        lw=0,
        facecolor=colors,
    )

    return collection

def plot_motifs(ax, motifdata, pwms):
    # do the actual plotting of motifs
    prev_max_xs = [] # keeps track of previously plotted x and y to avoid overlap
    y = 0
    full_max_y = 0 # keeps track of the maximum y to know dimensions of plot

    motifdata = motifdata.sort_values("position")

    for _, row in motifdata.iterrows():
        pwm = pwms[row["motif"]].numpy()
        length = pwm.shape[0]
        x = row["position"]
        max_x = row["position"] + length
        color = row["color"]
        label = row["label"]
        prev_max_xs = [(prev_max_x, y) for prev_max_x, y in prev_max_xs if x < prev_max_x]

        if len(prev_max_xs) > 0:
            ys = [y for _, y in prev_max_xs]
            for i in range(0, -10, -1):
                if i not in ys:
                    y = i
                    break
        else:
            y = 0

        full_max_y = max(y, full_max_y)

        rect = mpl.patches.Rectangle(
            (x, y - 1),
            length,
            1,
            fc = color,
            alpha = 0.1,
            zorder = -5,
        )
        ax.add_patch(rect)

        if row["strand"] == -1:
            pwm = pwm[::-1, ::-1]

        # plot motif
        collection = plot_motif(pwm, x, y-1)
        ax.add_collection(collection)

        # plot motif name
        text = ax.text(
            x,
            # x + length / 2,
            y-0.5,
            label,
            # ha="center",
            ha="right",
            va="center",
            fontsize=6,
            color = color,
            # color = "white",
            fontweight = "bold",
        )
        text.set_path_effects(
            [
                mpl.patheffects.withStroke(linewidth=1, foreground="white"),
            ]
        )

        prev_max_xs.append((max_x, y))

    return full_max_y-2

from .plot import _process_grouped_motifs



class GroupedMotifsGenomeBroken(Broken):
    def __init__(
        self,
        motifscan,
        gene,
        motifs_oi,
        breaking,
        genome,
        # group_info,
        pwms,
        panel_height=0.2,
    ):
        """
        Plot the location of motifs in a region.

        Parameters
        ----------
        
        """

        super().__init__(breaking = breaking, height = 0.2)

        # plot the motifs
        for (window, window_info), (panel, ax) in zip(
            breaking.regions.iterrows(), self
        ):
            ax.set_xlim(window_info["start"], window_info["end"])
            ax.axis("off")

            sequence = genome.fetch(
                window_info["chrom"], window_info["start_chrom"], window_info["end_chrom"]
            )

            collection = plot_sequence(sequence, window_info["start"])
            ax.add_collection(collection)

            motifdata = _process_grouped_motifs(
                gene,
                motifs_oi,
                motifscan,
                return_strands = True,
                window = [window_info["start"], window_info["end"]],
            )

            motifdata["label"] = motifscan.motifs.loc[motifdata["motif"]]["tf"].values
            motifdata["color"] = motifs_oi.reset_index().set_index(["group", "motif"])["color"].loc[pd.MultiIndex.from_frame(motifdata[["group", "motif"]])].values

            full_max_y = plot_motifs(ax, motifdata, pwms)

            ax.set_ylim(full_max_y, 1)

            ax.dim = (ax.width, (-full_max_y)*panel_height)
            

