---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# I don't trust peaks
# Fragment-based modeling of ATAC-seq data


Hypotheses:
- There is more information in ATAC-seq data than can be extracted from peaks
- Ideally, we would extract this information in a completely unbiased fashion, and not look at shapes
- Looking at this levels improves interpretation of the data and/or prediction of other modalities


## Previous literature


Barely any really... Weird!


There are "shape-based" differential expression methods, as discussed here (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-1929-3):

> Currently, most studies assume that ATAC-seq reads in peak regions follow a NB distribution, as is the case for RNA-seq data. However, no shape-based differential analysis tools exist for ATAC-seq data. The peaks contain not only read count information, but also the distribution shape profile. It is especially important for broad peaks, as broad peaks can contain multiple local maxima, and those shifts can indicate biologically relevant perturbations, which could be detected in sliding window or shape-based methods. Although not systematically studied, we believe incorporating shape information will improve differential peak analysis. Nevertheless, considering replicate handling, external peak caller dependency and backend statistical methods, csaw is worth a first try due to its easily explainable edgeR framework.

> Shape-based peak callers are not currently used in ATAC-seq, but they utilize read density profile information directly or indirectly and are believed to improve peak calling in ChIP-seq [73]. PICS [74] models fragment positions other than counts and calculate enrichment score for each candidate region. PolyaPeak [75] ranks peaks using statistics describing peak shape. CLC [76] learns a Gaussian filter for peak shape from positive and negative peaks.


There are methods that look at windows of ATAC-seq (and count individual cut sites): https://www.biorxiv.org/content/10.1101/2022.03.16.484118v1 Not really super exciting...


Some denoising of scATAC-seq data using RNNs: https://www.nature.com/articles/s41467-021-21765-5: Seems interesting at first, but actually isn't.


There is some literature that discusses how promoter architecture influences the stochasticity of the downstream mRNAs
- https://www.biorxiv.org/content/10.1101/2021.10.29.466407v2.full.pdf


Why do we actually aggregate over peaks? It's historical...
- From bulk
  - We've got a couple of samples => power is extremely low! There's not much more that you can detect except "this peak is higher in this condition"
  - However, with single-cell ATAC-seq, there are much more possibilities for mechanistic/biophysical insights
- From ChIP-Seq
  - Where a binding event is more likely to be discrete
  - However, for ATAC-seq, this is almost certainly not gonna be true
- From classical statistics/data analysis
  - Where modeling intervals on sequences is just not easy. Classical data analysis requires matrices
  - But, as was shown for images and sequences, if you create a "gradient-proof" pipeline, you can do anything
- Based on an assumption
  - There are relatively static open chromatin regions (=enhancers) and these are the units of gene regulation
  - Might be, might not be... Did anyone check this hypothesis?



All in all, it might be worth exploring how a "peak-free" ATAC-seq (and others...) modeling framework would look like.

In particular because in TF-seq we are considering looking at the earliest time points of gene regulation, and how TF binding / open chromatin changes. It would be a shame if we would miss the earliest causal events simply because we aggregate over peaks....


## Technology: what?
![image.png](atac_seq.png)


It's paired end, see for reasons: https://informatics.fas.harvard.edu/atac-seq-guidelines.html#:~:text=For%20ATAC%2Dseq%2C%20we%20recommend,less%20accessible%20to%20the%20assay.


## Maths: how?


The data:
- $X$: the ATAC-seq data
  - $X_\text{fragments}$ A dataframe containing chromosome, start, end and cell (barcode)
  - $X_\text{seq}$: A set of sequences that are inside a fragment, or around cut sites, for each cell
- $Y$: any other cell data, e.g.:
  - $Y_\text{exp}$: The expression data. Counts. ~NegativeBinomial.
  - $Y_\text{ct}$: Celltype labels for each cell. ~OneHotCategorical.
  - $Y_\text{diff}$: A differentiationt time for each cell. Single number. ~LogitNormal.
- $Z$: A latent space of the cells


Overall we have three use cases
- $X \rightarrow Y$: Predicting/classifying using ATAC-seq data
- $Y \rightarrow X$: Predicting/classifying the ATAC-seq data
- $X \rightarrow Z \rightarrow X$: Unsupervised learning of ATAC-seq data

The unsupervised learning task is essentially a combination of the two prediction tasks.


We could work at the following scales:
- Base-pair: model whether a base-pair is part of a fragment and in how much(?)
$$
\begin{align} 
P(X_\text{infragment}|\theta) &= P(locus,chr|\theta) \\
\end{align}
$$
- Cut sites: model where cuts are made (regardless of the other cut)
$$
\begin{align} 
P(X_\text{cut}|\theta) &= P(locus,chr|\theta) \\
\end{align}
$$
- Fragments: model the start and end cuts
$$
\begin{align} 
P(X_\text{fragments}|\theta) &= P(start, end,chr|\theta) \\
&= P(start, end|chr, \theta)P(chr|\theta) \\
&= P\left(end|start, chr, \theta\right)P(start|chr,\theta)P\left(chr|\theta\right) \\
\end{align}
$$


### $X\rightarrow Z$





### Biology: why?
