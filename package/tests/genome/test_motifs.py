import torch

from peakfreeatac.genome.motifs import score_fragments

class TestSimple():
    def test_single_gene(self):
        n_genes = 1
        genemapping = torch.tensor([0])
        coordinates = torch.tensor([[-50, 60]])

        window = (-100, 100)
        window_width = window[1] - window[0]
        cutwindow = (-10, 10)
        cutwindow_width = cutwindow[1] - cutwindow[0]

        motifscores_view = torch.zeros((window_width * n_genes, 1))
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert fragmentmotifscores.shape[0] == genemapping.shape[0]
        assert (fragmentmotifscores == 0).all()

        motifscores_view = torch.zeros((window_width * n_genes, 1))
        motifscores_view[40 + 0 * window_width, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores == 1/21).all()

        motifscores_view = torch.zeros((window_width * n_genes, 1))
        motifscores_view[39 + 0 * window_width, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores == 0).all()

        motifscores_view = torch.zeros((window_width * n_genes, 1))
        motifscores_view[50 + 0 * window_width, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores == 1/21).all()

        motifscores_view = torch.zeros((window_width * n_genes, 1))
        motifscores_view[60 + 0 * window_width, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores == 1/21).all()

        motifscores_view = torch.zeros((window_width * n_genes, 1))
        motifscores_view[61 + 0 * window_width, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores == 0.).all()
    
    def test_multiple_genes(self):
        n_genes = 2
        genemapping = torch.tensor([0, 1, 0])
        coordinates = torch.tensor([[-5, 6], [-10, 2], [-2, 9]])

        window = (-10, 10)
        window_width = window[1] - window[0]
        cutwindow = (-1, 1)
        cutwindow_width = cutwindow[1] - cutwindow[0]

        #
        motifscores_view = torch.zeros((window_width * n_genes, 2))
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores == 0.).all(), "All scores should be 0"

        #
        motifscores_view = torch.zeros((window_width * n_genes, 2))
        motifscores_view[5, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores[0, 0] == 1/3).all(), "Score at 0,0 should be 1/3"
        assert (fragmentmotifscores.flatten()[1:] == 0).all(), "Total score should be 1/3"

        #
        motifscores_view = torch.zeros((window_width * n_genes, 2))
        motifscores_view[1 + window_width, 0] = 1
        fragmentmotifscores = score_fragments(genemapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
        assert (fragmentmotifscores[1, 0] == 1/3).all(), "Score at 1,0 should be 1/3"
        assert (fragmentmotifscores.flatten().sum() == 1/3).all(), "Total score should be 1/3"