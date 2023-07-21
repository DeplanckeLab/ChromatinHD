class TestModel:
    def test_basic(self):
        fragments = []
        clustering = []
        fold = []
        model = []
        result = model.get_prediction(
            fragments, clustering, cell_ixs=fold["cells_test"]
        )
