import chromatinhd as chd
import numpy as np
import pandas as pd


class TestFragments:
    def test_example(self, example_fragments):
        loader = chd.loaders.fragments.Fragments(
            fragments=example_fragments,
            cellxregion_batch_size=10000,
        )

        minibatch = chd.loaders.minibatches.Minibatch(cells_oi=np.arange(20), regions_oi=np.arange(5), phase="train")
        data = loader.load(minibatch)

        assert data.coordinates.shape[0] == data.n_fragments
        assert data.coordinates.shape[0] == data.local_cellxregion_ix.shape[0]


class TestCuts:
    def test_example(self, example_fragments):
        loader = chd.loaders.fragments.Cuts(
            fragments=example_fragments,
            cellxregion_batch_size=10000,
        )

        minibatch = chd.loaders.minibatches.Minibatch(cells_oi=np.arange(20), regions_oi=np.arange(5), phase="train")
        data = loader.load(minibatch)

        assert data.coordinates.shape[0] == data.n_cuts
        assert data.coordinates.shape[0] == data.local_cellxregion_ix.shape[0]


class TestFragmentsView:
    def test_tiny(self):
        obs = pd.DataFrame(
            {
                "cell": ["C0", "C1", "C2"],
            }
        ).set_index("cell")

        parentregion_coordinates = pd.DataFrame({"chrom": ["a", "b"], "start": [0, 0], "end": [100, 100]}).set_index(
            "chrom", drop=False
        )
        parentregions = chd.data.Regions.create(coordinates=parentregion_coordinates)

        region_coordinates = pd.DataFrame(
            {
                "chrom": ["a", "a", "b"],
                "start": [0, 50, 50],
                "end": [60, 110, 110],
                "region": ["a1", "a2", "b1"],
                "strand": [1, 1, -1],
            }
        ).set_index("region")
        regions = chd.data.Regions.create(coordinates=region_coordinates, window=np.array([0, 60]))

        parentfragments = chd.data.Fragments.create(
            regions=parentregions,
            coordinates=np.array(
                [
                    [11, 20],
                    [15, 21],
                    [41, 59],
                    [51, 71],
                    [1, 20],
                    [1, 30],
                    [53, 60],
                    [52, 61],
                    [70, 72],
                    [51, 71],
                ],
                dtype=np.int32,
            ),
            mapping=np.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 0],
                    [2, 0],
                    [0, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [2, 1],
                ],
                dtype=np.int32,
            ),
            obs=obs,
            var=parentregion_coordinates,
        )
        fragments = chd.data.fragments.FragmentsView.from_fragments(parentfragments, regions, overwrite=True)
        fragments.create_regionxcell_indptr()

        # test regionxcell_indptr
        for region_ix in range(regions.n_regions):
            region = regions.coordinates.iloc[region_ix]
            parentregion_ix = parentregions.coordinates.index.get_loc(region["chrom"])
            for cell_ix in range(obs.shape[0]):
                fragments_oi = (
                    (parentfragments.mapping[:, 0] == cell_ix)
                    & (parentfragments.mapping[:, 1] == parentregion_ix)
                    & (parentfragments.coordinates[:, 0] >= region["start"])
                    & (parentfragments.coordinates[:, 0] < region["end"])
                )

                regionxcell_ix = region_ix * fragments.n_cells + cell_ix
                indptr_start, indptr_end = fragments.regionxcell_indptr[regionxcell_ix, :]

                fragment_ixs = np.arange(indptr_start, indptr_end)

                assert fragments_oi.sum() == len(fragment_ixs)
                assert np.all(fragment_ixs == np.where(fragments_oi)[0])

        # test loader
        loader = chd.loaders.Fragments(fragments, cellxregion_batch_size=10)

        assert (loader.region_centers == np.array([0, 50, 110])).all()
        for region_ix in range(regions.n_regions):
            region = regions.coordinates.iloc[region_ix]
            parentregion_ix = parentregions.coordinates.index.get_loc(region["chrom"])
            for cell_ix in range(obs.shape[0]):
                fragments_oi = (
                    (parentfragments.mapping[:, 0] == cell_ix)
                    & (parentfragments.mapping[:, 1] == parentregion_ix)
                    & (parentfragments.coordinates[:, 0] >= region["start"])
                    & (parentfragments.coordinates[:, 0] < region["end"])
                )

                regionxcell_ix = region_ix * fragments.n_cells + cell_ix
                indptr_start, indptr_end = fragments.regionxcell_indptr[regionxcell_ix, :]

                data = loader.load(
                    chd.loaders.minibatches.Minibatch(cells_oi=np.array([cell_ix]), regions_oi=np.array([region_ix]))
                )

                assert data.n_fragments == fragments_oi.sum(), f"{region_ix=} {parentregion_ix=} {cell_ix=}"
                expected = (
                    (fragments.coordinates[fragments_oi])
                    - (region["start"] - fragments.regions.window[0]) * (region["strand"] == 1).astype(int)
                    - (region["end"] - fragments.regions.window[0]) * (region["strand"] == -1).astype(int)
                ) * region["strand"]
                found = data.coordinates.numpy()
                assert np.allclose(found, expected)
