import pandas as pd
import urllib
import pathlib
import chromatinhd.data.motifscan
import json


def get_hocomoco_11(path, organism="human", variant="core", overwrite=False):
    """
    Download hocomoco human data

    Parameters:
        path:
            the path to download to
        organism:
            the organism to download for, either "human" or "mouse"
        variant:
            the variant to download, either "full" or "core"
        overwrite:
            whether to overwrite existing files
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
    if overwrite or (not (path / "pwm_cutoffs.txt").exists()):
        urllib.request.urlretrieve(
            f"https://hocomoco11.autosome.org/final_bundle/hocomoco11/{variant}/{organism}/mono/HOCOMOCOv11_{variant}_standard_thresholds_{organism}_mono.txt",
            path / "pwm_cutoffs.txt",
        )
        urllib.request.urlretrieve(
            f"https://hocomoco11.autosome.org/final_bundle/hocomoco11/{variant}/{organism}/mono/HOCOMOCOv11_{variant}_pwms_{organism}_mono.txt",
            path / "pwms.txt",
        )
        urllib.request.urlretrieve(
            f"https://hocomoco11.autosome.org/final_bundle/hocomoco11/{variant}/{organism}/mono/HOCOMOCOv11_{variant}_annotation_{organism}_mono.tsv",
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


def get_hocomoco(path, organism="hs", variant="CORE", overwrite=False):
    """
    Download hocomoco human data

    Parameters:
        path:
            the path to download to
        organism:
            the organism to download for, either "hs" or "mm"
        variant:
            the variant to download, either "INVIVO" or "CORE"
        overwrite:
            whether to overwrite existing files
    """
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # download cutoffs, pwms and annotations
    if overwrite or (not (path / "pwm_cutoffs.txt").exists()):
        urllib.request.urlretrieve(
            f"https://hocomoco12.autosome.org/final_bundle/hocomoco12/H12{variant}/H12{variant}_annotation.jsonl",
            path / "annotation.jsonl",
        )
        urllib.request.urlretrieve(
            f"https://hocomoco12.autosome.org/final_bundle/hocomoco12/H12{variant}/H12{variant}_pwm.tar.gz",
            path / "pwms.tar.gz",
        )
        urllib.request.urlretrieve(
            f"https://hocomoco12.autosome.org/final_bundle/hocomoco12/H12{variant}/H12{variant}_thresholds.tar.gz",
            path / "thresholds.tar.gz",
        )

    pwms = chromatinhd.data.motifscan.read_pwms(path / "pwms.tar.gz")
    motifs = [json.loads(line) for line in open(path / "annotation.jsonl").readlines()]
    motifs = pd.DataFrame(motifs).set_index("name")
    motifs.index.name = "motif"

    for thresh in motifs["standard_thresholds"].iloc[0].keys():
        motifs["cutoff_" + thresh] = [thresholds[thresh] for _, thresholds in motifs["standard_thresholds"].items()]
    for species in ["HUMAN", "MOUSE"]:
        motifs[species + "_gene_symbol"] = [
            masterlist_info["species"][species]["gene_symbol"] if species in masterlist_info["species"] else None
            for _, masterlist_info in motifs["masterlist_info"].items()
        ]
    motifs["symbol"] = motifs["HUMAN_gene_symbol"] if organism == "hs" else motifs["MOUSE_gene_symbol"]

    return pwms, motifs
