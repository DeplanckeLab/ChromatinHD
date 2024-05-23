import chromatinhd as chd
import time


class TestPred:
    def test_example_single_model(self, example_fragments, example_transcriptome, example_folds):
        fold = example_folds[0]
        model = chd.models.pred.model.better.Model.create(
            fragments=example_fragments, transcriptome=example_transcriptome, region_oi=example_fragments.var.index[0]
        )

        start = time.time()
        model.train_model(fold=example_folds[0], n_epochs=10)

        delta = time.time() - start

        assert delta < 20
        result = model.get_prediction(cell_ixs=fold["cells_test"])

        assert result["predicted"].shape[0] == len(fold["cells_test"])

    def test_example_multiple_models(self, example_fragments, example_transcriptome, example_folds):
        model = chd.models.pred.model.better.Models.create(
            fragments=example_fragments, transcriptome=example_transcriptome
        )

        start = time.time()
        model.train_models(folds=folds, n_epochs=10)

        delta = time.time() - start

        assert delta < 20
        result = model.get_prediction(cell_ixs=fold["cells_test"])

        assert result["predicted"].shape[0] == len(fold["cells_test"])
