import pandas as pd
import numpy as np


def group_enrichment(
    enrichment,
    slicecounts,
    clustering,
    merge_cutoff=0.2,
    q_value_cutoff=0.01,
    odds_cutoff=1.1,
    min_found=100,
):
    """
    Group motifs based on correlation of their slice counts across clusters

    Parameters
    ----------
    enrichment : pd.DataFrame
        DataFrame with columns "odds", "q_value", "contingency"
    slicecounts : pd.DataFrame
        DataFrame with slice counts
    clustering : pd.DataFrame
        DataFrame with cluster information
    merge_cutoff : float
        Correlation cutoff for merging motifs
    q_value_cutoff : float
        q-value cutoff for enrichment
    odds_cutoff : float
        Odds ratio cutoff for enrichment
    min_found : int
        Minimum number of found sites for enrichment
    """

    enrichment_grouped = []
    for cluster_oi in clustering.cluster_info.index:
        slicecors = pd.DataFrame(
            np.corrcoef((slicecounts.T > 0) + np.random.normal(0, 1e-6, slicecounts.shape).T),
            index=slicecounts.columns,
            columns=slicecounts.columns,
        )
        enrichment["found"] = enrichment["contingency"].map(lambda x: x[1, 1].sum())
        enrichment_oi = (
            enrichment.loc[cluster_oi]
            .query("q_value < @q_value_cutoff")
            .query("odds > @odds_cutoff")
            .sort_values("odds", ascending=False)
            .query("found > @min_found")
        )
        # enrichment_oi = enrichment_oi.loc[(~enrichment_oi.index.get_level_values("motif").str.contains("ZNF")) & (~enrichment_oi.index.get_level_values("motif").str.startswith("ZN")) & (~enrichment_oi.index.get_level_values("motif").str.contains("KLF")) & (~enrichment_oi.index.get_level_values("motif").str.contains("WT"))]
        motif_grouping = {}
        for motif_id in enrichment_oi.index:
            slicecors_oi = slicecors.loc[motif_id, list(motif_grouping.keys())]
            if (slicecors_oi < merge_cutoff).all():
                motif_grouping[motif_id] = [motif_id]
                enrichment_oi.loc[motif_id, "group"] = motif_id
            else:
                group = slicecors_oi.sort_values(ascending=False).index[0]
                motif_grouping[group].append(motif_id)
                enrichment_oi.loc[motif_id, "group"] = group
        enrichment_group = enrichment_oi.sort_values("odds", ascending=False).loc[
            list(motif_grouping.keys())
        ]
        enrichment_group["members"] = [
            motif_grouping[group] for group in enrichment_group.index
        ]
        enrichment_grouped.append(enrichment_group.assign(cluster=cluster_oi))
    enrichment_grouped = (
        pd.concat(enrichment_grouped).reset_index().set_index(["cluster", "motif"])
    )
    enrichment_grouped = enrichment_grouped.sort_values("q_value", ascending=True)

    return enrichment_grouped
