def copy_data(folder:pathlib.Path):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["fragments", "transcriptome"]:
        shutil.copytree(
            src=pathlib.Path("src/chromatinhd/data/example/tiny") / name,
            dst=folder / name,
        )