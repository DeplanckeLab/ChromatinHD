import chromatinhd as chd
import time


class TestAdditive:
    def test_example(self, example_fragments, example_transcriptome, example_folds):
        fold = example_folds[0]
        model = chd.models.pred.model.additive.Model(
            n_genes=len(example_fragments.var.index)
        )

        start = time.time()
        model.train_model(
            example_fragments, example_transcriptome, fold=fold, n_epochs=1
        )

        delta = time.time() - start

        assert delta < 20
        result = model.get_prediction(
            example_fragments, example_transcriptome, cell_ixs=fold["cells_test"]
        )

        assert result["predicted"].shape[0] == len(fold["cells_test"])
