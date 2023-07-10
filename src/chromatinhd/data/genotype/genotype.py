from chromatinhd.flow import Flow, Stored


class Genotype(Flow):
    genotypes = Stored("genotypes")
    variants_info = Stored("variants_info")
