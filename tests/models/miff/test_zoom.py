import chromatinhd.models.miff.model.zoom


class TestZoom:
    def test_simple(self):
        width = 1024
        positions = torch.tensor([100, 500, 1000])
        nbins = [8, 8, 8]

        unnormalized_heights_bins = [
            torch.tensor([[1] * 8] * 3, dtype=torch.float),
            torch.tensor([[1] * 8] * 3, dtype=torch.float),
            torch.tensor([[1] * 8] * 3, dtype=torch.float),
        ]
        unnormalized_heights_bins[1][1, 3] = 10
