<p align="center">
  <a href="https://chromatinhd.eu">
    <img src="https://raw.githubusercontent.com/DeplanckeLab/ChromatinHD/main/docs/source/static/logo.png" width="300" />
  </a>
  <a href="https://chromatinhd.eu">
    <img src="https://raw.githubusercontent.com/DeplanckeLab/ChromatinHD/main/docs/source/static/comparison.gif" />
  </a>
</p>

ChromatinHD analyzes single-cell ATAC+RNA data using the raw fragments as input,
by automatically adapting the scale at which
relevant chromatin changes on a per-position, per-cell, and per-gene basis.
This enables identification of functional chromatin changes
regardless of whether they occur in a narrow or broad region.

As we show in [our paper](https://www.biorxiv.org/content/10.1101/2023.07.21.549899v1):
- Compared to the typical approach (peak calling + statistical analysis), ChromatinHD models are better able to capture functional chromatin changes. This is because there are extensive functional accessibility changes both outside and within peaks ([Figure 3](https://www.biorxiv.org/content/10.1101/2023.07.21.549899v1)).
- ChromatinHD models can capture long-range interactions by considering fragments co-occuring within the same cell ([Figure 4](https://www.biorxiv.org/content/10.1101/2023.07.21.549899v1)).
- ChromatinHD models can also capture changes in fragment size that are related to gene expression changes, likely driven by dense direct and indirect binding of transcription factors ([Figure 5](https://www.biorxiv.org/content/10.1101/2023.07.21.549899v1)).

[📜 Manuscript](https://www.biorxiv.org/content/10.1101/2023.07.21.549899v1)

[❔ Documentation](https://chromatinhd.eu)

[▶️ Quick start](https://chromatinhd.eu/quickstart/0_install)
