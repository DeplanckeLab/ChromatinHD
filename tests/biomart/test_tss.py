import chromatinhd as chd
class TestGetCanonicalTranscripts():
    def test_simple(self, dataset_grch38):
        transcripts = chd.biomart.tss.get_canonical_transcripts(dataset = dataset_grch38)

        assert transcripts.shape[0] > 1000


class TestGetExons():
    def test_simple(self, dataset_grch38):
        exons = chd.biomart.tss.get_exons(dataset = dataset_grch38)

        assert exons.shape[0] > 1000

