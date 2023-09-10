from .additive import Model as ModelBase


class Model(ModelBase):
    def __init__(self, window, *args, **kwargs):
        self.window = window
        super().__init__(*args, **kwargs)

    def forward(self, data):
        """
        Make a prediction given a data object
        """
        if self.window is not None:
            window = self.window

            coordinates = data.fragments.coordinates
            window_start = window[0]
            window_end = window[1]

            selected_fragments = ~(
                ((coordinates[:, 0] < window_end) & (coordinates[:, 0] > window_start))
                | ((coordinates[:, 1] < window_end) & (coordinates[:, 1] > window_start))
            )

            data.fragments.coordinates = data.fragments.coordinates[selected_fragments]
            data.fragments.regionmapping = data.fragments.regionmapping[selected_fragments]
            data.fragments.local_cellxregion_ix = data.fragments.local_cellxregion_ix[selected_fragments]

        return super().forward(data)
