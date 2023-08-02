import chromatinhd as chd


class TestGetCanonicalTranscripts:
    def test_simple(self, dataset_grch38):
        pass
        # transcripts = chd.biomart.tss.get_canonical_transcripts(biomart_dataset = dataset_grch38)

        # assert transcripts.shape[0] > 1000


class TestGetExons:
    def test_simple(self, dataset_grch38):
        pass
        # exons = chd.biomart.tss.get_exons(biomart_dataset = dataset_grch38, chrom = "chr1", start = 1000, end = 2000)
