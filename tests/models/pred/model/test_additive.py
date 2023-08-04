import chromatinhd as chd


class TestAdditive:
    def test_example(self, example_fragments, example_transcriptome, example_folds):
        model = chd.models.pred.model.additive.Model(
            n_genes=len(example_fragments.var.index)
        )
        model.train_model(
            example_fragments, example_transcriptome, fold=example_folds[0], n_epochs=1
        )
        # result = model.get_prediction(
        #     fragments, clustering, cell_ixs=fold["cells_test"]
        # )
