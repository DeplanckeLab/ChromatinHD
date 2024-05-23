import chromatinhd as chd
import time


class TestDiff:
    def test_example_single_model(self, example_fragments, example_clustering, example_folds):
        fold = example_folds[0]
        model = chd.models.diff.model.binary.Model.create(
            fragments=example_fragments, clustering=example_clustering, fold=fold
        )

        start = time.time()
        model.train_model(n_epochs=10)

        delta = time.time() - start

        assert delta < 20

    def test_example_multiple_models(self, example_fragments, example_clustering, example_folds):
        model = chd.models.diff.model.binary.Models.create(
            fragments=example_fragments,
            clustering=example_clustering,
            folds=example_folds,
        )

        start = time.time()
        model.train_models(n_epochs=2, regions_oi=example_fragments.var.index[:2])

        delta = time.time() - start

        assert delta < 40
