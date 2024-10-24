import chromatinhd.models.miff.model.zoom
import torch
import numpy as np


class TestZoom:
    def test_simple(self):
        width = 16
        nbins = [4, 2, 2]

        totalnbins = np.cumprod(nbins)
        totalbinwidths = torch.tensor(width // totalnbins)

        positions = torch.arange(16)

        def test_probability(unnormalized_heights_all, expected_logprob):
            totalnbins = np.cumprod(nbins)
            totalbinwidths = torch.tensor(width // totalnbins)

            unnormalized_heights_zooms = chromatinhd.models.miff.model.zoom.extract_unnormalized_heights(
                positions, totalbinwidths, unnormalized_heights_all
            )
            logprob = chromatinhd.models.miff.model.zoom.calculate_logprob(
                positions, nbins, width, unnormalized_heights_zooms
            )

            # print(1 / np.exp((logprob)))
            assert np.isclose(1 / np.exp((logprob)), expected_logprob).all()

        ##
        unnormalized_heights_all = []
        cur_total_n = 1
        for n in nbins:
            cur_total_n *= n
            unnormalized_heights_all.append(torch.zeros(cur_total_n).reshape(-1, n))
        unnormalized_heights_all[1][2, 1] = np.log(2)
        expected_logprob = np.array([16, 16, 16, 16, 16, 16, 16, 16, 24, 24, 12, 12, 16, 16, 16, 16])

        test_probability(unnormalized_heights_all, expected_logprob)

        ##
        unnormalized_heights_all = []
        cur_total_n = 1
        for n in nbins:
            cur_total_n *= n
            unnormalized_heights_all.append(torch.zeros(cur_total_n).reshape(-1, n))
        unnormalized_heights_all[0][0, 1] = np.log(2)
        expected_logprob = np.array([20, 20, 20, 20, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20])

        test_probability(unnormalized_heights_all, expected_logprob)

        ##
        unnormalized_heights_all = []
        cur_total_n = 1
        for n in nbins:
            cur_total_n *= n
            unnormalized_heights_all.append(torch.zeros(cur_total_n).reshape(-1, n))
        unnormalized_heights_all[2][2, 0] = np.log(2)
        expected_logprob = np.array([16, 16, 16, 16, 12, 24, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16])

        test_probability(unnormalized_heights_all, expected_logprob)
