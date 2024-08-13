import numpy as np
import scanpy as sc
import pandas as pd


# Define cluster score for all markers
def evaluate_partition(anndata, marker_dict, gene_symbol_key=None, partition_key="louvain_r1"):
    # Inputs:
    #    anndata         - An AnnData object containing the data set and a partition
    #    marker_dict     - A dictionary with cell-type markers. The markers should be stores as anndata.var_names or
    #                      an anndata.var field with the key given by the gene_symbol_key input
    #    gene_symbol_key - The key for the anndata.var field with gene IDs or names that correspond to the marker
    #                      genes
    #    partition_key   - The key for the anndata.obs field where the cluster IDs are stored. The default is
    #                      'louvain_r1'

    # Test inputs
    if partition_key not in anndata.obs.columns.values:
        print("KeyError: The partition key was not found in the passed AnnData object.")
        print("   Have you done the clustering? If so, please tell pass the cluster IDs with the AnnData object!")
        raise

    if (gene_symbol_key is not None) and (gene_symbol_key not in anndata.var.columns.values):
        print("KeyError: The provided gene symbol key was not found in the passed AnnData object.")
        print("   Check that your cell type markers are given in a format that your anndata object knows!")
        raise

    if gene_symbol_key:
        gene_ids = anndata.var[gene_symbol_key]
    else:
        gene_ids = anndata.var_names

    clusters = np.unique(anndata.obs[partition_key])
    n_clust = len(clusters)
    n_groups = len(marker_dict)

    marker_res = np.zeros((n_groups, n_clust))
    z_scores = sc.pp.scale(anndata, copy=True)

    i = 0
    for group in marker_dict:
        # Find the corresponding columns and get their mean expression in the cluster
        j = 0
        for clust in clusters:
            cluster_cells = np.in1d(z_scores.obs[partition_key], clust)
            marker_genes = np.in1d(gene_ids, marker_dict[group])
            marker_res[i, j] = z_scores.X[np.ix_(cluster_cells, marker_genes)].mean()
            j += 1
        i += 1

    variances = np.nanvar(marker_res, axis=0)
    if np.all(np.isnan(variances)):
        print("No variances could be computed, check if your cell markers are in the data set.")
        print("Maybe the cell marker IDs do not correspond to your gene_symbol_key input or the var_names")
        raise

    marker_res_df = pd.DataFrame(marker_res, columns=clusters, index=marker_dict.keys())

    # Return the median of the variances over the clusters
    return marker_res_df
