import torch
import numpy as np
import pandas as pd


def count_gc(relative_starts, relative_end, gene_ixs, onehot_promoters, window):
    eps = 1e-5
    gc = []
    for relative_start, relative_end, gene_ix in zip(
        relative_starts, relative_end, gene_ixs
    ):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        gc.append(
            onehot_promoters[start_ix:end_ix, [1, 2]].sum() / (end_ix - start_ix + eps)
        )

    gc = torch.hstack(gc).numpy()[:, None]

    return gc


def count_dinuc(relative_starts, relative_end, gene_ixs, onehot_promoters, window):
    gc = []
    eps = 1e-5
    for relative_start, relative_end, gene_ix in zip(
        relative_starts, relative_end, gene_ixs
    ):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        nuc = torch.where(onehot_promoters[start_ix:end_ix])[1]
        gc.append(
            torch.bincount(nuc[1:] + (nuc[:-1] * 4), minlength=16)
            / (end_ix - start_ix + eps)
        )

    gc = torch.stack(gc).numpy()

    return gc


def count_motifs(relative_starts, relative_end, gene_ixs, motifscan, window):
    motif_indices = []
    for relative_start, relative_end, gene_ix in zip(
        relative_starts, relative_end, gene_ixs
    ):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        motif_indices.append(
            motifscan.indices[motifscan.indptr[start_ix] : motifscan.indptr[end_ix]]
        )
    motif_indices = np.hstack(motif_indices)
    motif_counts = np.bincount(motif_indices, minlength=motifscan.n_motifs)

    return motif_counts


def count_motifs_genewise(
    relative_starts, relative_end, gene_ixs, motifscan, n_genes, window
):
    motif_indices = []
    for relative_start, relative_end, gene_ix in zip(
        relative_starts, relative_end, gene_ixs
    ):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        motif_indices.append(
            motifscan.indices[motifscan.indptr[start_ix] : motifscan.indptr[end_ix]]
            + gene_ix * motifscan.n_motifs
        )
    motif_indices = np.hstack(motif_indices)
    motif_counts = np.bincount(
        motif_indices, minlength=motifscan.n_motifs * n_genes
    ).reshape((n_genes, motifscan.n_motifs))

    return motif_counts


def count_motifs_slicewise(relative_starts, relative_end, gene_ixs, motifscan, window):
    motif_indices = []
    n_slices = len(relative_starts)
    for relative_start, relative_end, gene_ix, slice_ix in zip(
        relative_starts, relative_end, gene_ixs, range(n_slices)
    ):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        motif_indices.append(
            motifscan.indices[motifscan.indptr[start_ix] : motifscan.indptr[end_ix]]
            + slice_ix * motifscan.n_motifs
        )
    motif_indices = np.hstack(motif_indices)
    motif_counts = np.bincount(
        motif_indices, minlength=motifscan.n_motifs * n_slices
    ).reshape((n_slices, motifscan.n_motifs))

    return motif_counts


def select_background(
    position_slices,
    gene_ixs_slices,
    onehot_promoters,
    window,
    n_genes,
    n_random=100,
    n_select_random=10,
    seed=None,
):
    # window_oi_gc = count_gc(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, onehot_promoters)
    window_oi_gc = count_dinuc(
        position_slices[:, 0],
        position_slices[:, 1],
        gene_ixs_slices,
        onehot_promoters,
        window,
    )

    if seed is not None:
        rg = np.random.RandomState(seed)
    else:
        rg = np.random.RandomState()

    # random position
    position_slices_repeated = position_slices.repeat(n_random, 0)
    random_position_slices = np.zeros_like(position_slices_repeated)
    random_position_slices[:, 0] = rg.randint(
        np.ones(position_slices_repeated.shape[0]) * window[0],
        np.ones(position_slices_repeated.shape[0]) * window[1]
        - (position_slices_repeated[:, 1] - position_slices_repeated[:, 0]),
    )
    random_position_slices[:, 1] = random_position_slices[:, 0] + (
        position_slices_repeated[:, 1] - position_slices_repeated[:, 0]
    )
    # random_position_slices = position_slices_repeated

    # random gene
    random_gene_ixs_slices = rg.randint(n_genes, size=random_position_slices.shape[0])
    # random_gene_ixs_slices = gene_ixs_slices.repeat(n_random, 0)

    # window_random_gc = count_gc(random_position_slices[:, 0], random_position_slices[:, 1], random_gene_ixs_slices, onehot_promoters)
    window_random_gc = count_dinuc(
        random_position_slices[:, 0],
        random_position_slices[:, 1],
        random_gene_ixs_slices,
        onehot_promoters,
        window,
    )

    random_difference = np.sqrt(
        (
            (
                window_random_gc.reshape(
                    (position_slices.shape[0], n_random, window_random_gc.shape[-1])
                )
                - window_oi_gc[:, None, :]
            )
            ** 2
        ).mean(-1)
    )

    chosen_background = np.argsort(random_difference, axis=1)[
        :, :n_select_random
    ].flatten()
    chosen_background_idx = (
        np.repeat(np.arange(position_slices.shape[0]), n_select_random) * n_random
        + chosen_background
    )

    # control
    # fig, ax = plt.subplots()
    # ax.scatter(window_oi_gc[:, 0], window_random_gc[::n_random, 0])
    # ax.scatter(window_oi_gc[:, 0], window_random_gc[chosen_background_idx][::n_select_random, 0])
    #

    background_position_slices = random_position_slices[chosen_background_idx]
    background_gene_ixs_slices = random_gene_ixs_slices[chosen_background_idx]

    # control
    # gc_back = count_gc(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, onehot_promoters)
    # ax.scatter(window_oi_gc[:, 13], gc_back[::n_select_random, 0])
    #

    # control 2
    # window_oi_dinuc = count_dinuc(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, onehot_promoters)
    # dinuc_back = count_dinuc(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, onehot_promoters)
    # ax.scatter(window_oi_dinuc[:, 13], dinuc_back[::n_select_random][:, 13])

    return background_position_slices, background_gene_ixs_slices


import fisher
import statsmodels.stats.multitest
import scipy.stats

# pip install fisher
def enrich_windows(
    motifscan,
    position_slices,
    gene_ixs_slices,
    n_genes,
    window,
    onehot_promoters=None,
    gene_ids=None,
    oi_slices=None,
    background_slices=None,
    n_background=10,
):
    background_position_slices = None
    if background_slices is not None:
        background_position_slices = position_slices[background_slices]
        background_gene_ixs_slices = gene_ixs_slices[background_slices]
    if oi_slices is not None:
        position_slices = position_slices[oi_slices]
        gene_ixs_slices = gene_ixs_slices[oi_slices]

    print(background_position_slices.shape, position_slices.shape)

    motif_counts = count_motifs(
        position_slices[:, 0],
        position_slices[:, 1],
        gene_ixs_slices,
        motifscan,
        window=window,
    )
    n_positions = (position_slices[:, 1] - position_slices[:, 0]).sum()
    # motif_counts_slicewise = count_motifs_slicewise(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, motifscan.indices, motifscan.indptr, motifscan.n_motifs)
    motif_counts_genewise = count_motifs_genewise(
        position_slices[:, 0],
        position_slices[:, 1],
        gene_ixs_slices,
        motifscan,
        n_genes,
        window,
    )
    n_positions_gene = np.bincount(
        gene_ixs_slices,
        weights=position_slices[:, 1] - position_slices[:, 0],
        minlength=n_genes,
    )
    motif_percs_genewise = motif_counts_genewise / (n_positions_gene[:, None] + 1e-5)

    if background_position_slices is None:
        background_position_slices, background_gene_ixs_slices = select_background(
            position_slices,
            gene_ixs_slices,
            onehot_promoters,
            seed=1,
            n_genes=n_genes,
            window=window,
            n_random=n_background * 10,
            n_select_random=n_background,
        )

    background_motif_counts = count_motifs(
        background_position_slices[:, 0],
        background_position_slices[:, 1],
        background_gene_ixs_slices,
        motifscan,
        window=window,
    )
    background_n_positions = (
        background_position_slices[:, 1] - background_position_slices[:, 0]
    ).sum()
    # motif_counts_slicewise2 = count_motifs_slicewise(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, motifscan.indices, motifscan.indptr, motifscan.n_motifs)

    # create contingencies to calculate conditional odds
    contingencies = (
        np.stack(
            [
                np.stack(
                    [
                        background_n_positions - background_motif_counts,
                        background_motif_counts,
                    ]
                ),
                np.stack([n_positions - motif_counts, motif_counts]),
            ]
        )
        .transpose(2, 0, 1)
        .astype(np.uint)
    )

    # odds_conditional = []
    # for cont in contingencies:
    #     odds_conditional.append(
    #         scipy.stats.contingency.odds_ratio(cont + 1, kind="conditional").statistic
    #     )  # pseudocount = 1

    p_values = fisher.pvalue_npy(
        contingencies[:, 0, 0],
        contingencies[:, 1, 0],
        contingencies[:, 0, 1],
        contingencies[:, 1, 1],
    )[-1]

    n_motifs = np.bincount(motifscan.indices, minlength=motifscan.n_motifs)

    # create motifscores
    motifscores = pd.DataFrame(
        {
            "odds": ((motif_counts + 1) / (n_positions + 1))
            / ((background_motif_counts + 1) / (background_n_positions + 1)),
            # "odds_conditional": odds_conditional,
            "motif": motifscan.motifs.index,
            "in": motif_counts / n_motifs,
            "perc": motif_counts / n_positions,
            "pval": p_values,
            "contingency": [cont for cont in contingencies],
            "perc_gene": [x for x in motif_percs_genewise.T],
        }
    ).set_index("motif")
    # motifscores["logodds"] = np.log(odds_conditional)
    motifscores["logodds"] = np.log(motifscores["odds"])
    motifscores["qval"] = statsmodels.stats.multitest.fdrcorrection(
        motifscores["pval"]
    )[-1]

    return motifscores


def detect_windows(motifscan, position_slices, gene_ixs_slices, gene_ids, window):
    motif_counts = count_motifs_genewise(
        position_slices[:, 0],
        position_slices[:, 1],
        gene_ixs_slices,
        motifscan,
        len(gene_ids),
        window,
    )
    gene_counts = (
        pd.DataFrame(
            {
                "n_positions": position_slices[:, 1] - position_slices[:, 0],
                "gene_ix": gene_ixs_slices,
            }
        )
        .groupby("gene_ix")["n_positions"]
        .sum()
    )
    gene_counts.index = gene_ids[gene_counts.index].values
    gene_counts = gene_counts.reindex(gene_ids, fill_value=0)

    motifscores = (
        pd.DataFrame(
            motif_counts,
            index=gene_ids,
            columns=motifscan.motifs.index,
        )
        .stack()
        .to_frame(name="n_positions")
    )
    motifscores["n_positions"] = gene_counts.loc[
        motifscores.index.get_level_values("gene")
    ].values

    return motifscores
