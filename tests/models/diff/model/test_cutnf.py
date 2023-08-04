import chromatinhd as chd
import time


class TestAdditive:
    def test_example(self, example_fragments, example_clustering, example_folds):
        fold = example_folds[0]
        model = chd.models.diff.model.cutnf.Model(example_fragments, example_clustering)

        start = time.time()
        model.train_model(example_fragments, example_clustering, fold=fold, n_epochs=1)

        delta = time.time() - start

        assert delta < 20
        result = model.get_prediction(
            example_fragments, example_clustering, cell_ixs=fold["cells_test"]
        )

        assert result["likelihood_mixture"].shape[0] == len(fold["cells_test"])
