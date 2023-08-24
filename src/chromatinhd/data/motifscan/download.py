import pandas as pd
import urllib
import pathlib
import chromatinhd.data.motifscan


def get_hocomoco(path, organism="human"):
    """
    Download hocomoco human data

    Parameters:
        path:
            the path to download to
        organism:
            the organism to download for, either "human" or "mouse"
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if organism == "human":
        organism = "HUMAN"
    elif organism == "mouse":
        organism = "MOUSE"
    else:
        raise ValueError(f"Unknown organism: {organism}")

    # download cutoffs, pwms and annotations
    if not (path / "pwm_cutoffs.txt").exists():
        urllib.request.urlretrieve(
            f"https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/{organism}/mono/HOCOMOCOv11_core_standard_thresholds_{organism}_mono.txt",
            path / "pwm_cutoffs.txt",
        )
        urllib.request.urlretrieve(
            f"https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/{organism}/mono/HOCOMOCOv11_core_pwms_{organism}_mono.txt",
            path / "pwms.txt",
        )
        urllib.request.urlretrieve(
            f"https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/{organism}/mono/HOCOMOCOv11_core_annotation_{organism}_mono.tsv",
            path / "annot.txt",
        )

    pwms = chromatinhd.data.motifscan.read_pwms(path / "pwms.txt")

    motifs = pd.DataFrame({"motif": pwms.keys()}).set_index("motif")
    motif_cutoffs = pd.read_table(
        path / "pwm_cutoffs.txt",
        names=["motif", "cutoff_001", "cutoff_0005", "cutoff_0001"],
        skiprows=1,
    ).set_index("motif")
    motifs = motifs.join(motif_cutoffs)
    annot = (
        pd.read_table(path / "annot.txt")
        .rename(columns={"Model": "motif", "Transcription factor": "gene_label"})
        .set_index("motif")
    )
    motifs = motifs.join(annot)

    return pwms, motifs
