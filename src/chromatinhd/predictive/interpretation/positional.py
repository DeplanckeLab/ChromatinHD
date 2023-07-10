class Positional:
    def calculate(self, plotdata, window_sizes_info, window):
        positions_oi = np.arange(*window)

        deltacor_test_interpolated = np.zeros(
            (len(window_sizes_info), len(positions_oi))
        )
        deltacor_validation_interpolated = np.zeros(
            (len(window_sizes_info), len(positions_oi))
        )
        retained_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        lost_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        for window_size, window_size_info in window_sizes_info.iterrows():
            # for window_size, window_size_info in window_sizes_info.query(
            #     "window_size == 200"
            # ).iterrows():
            plotdata_oi = plotdata.query("phase in ['validation']").query(
                "window_size == @window_size"
            )

            x = plotdata_oi["window_mid"].values.copy()
            y = plotdata_oi["deltacor"].values.copy()
            y[plotdata_oi["qval"] > 0.1] = 0.0
            deltacor_interpolated_ = np.clip(
                np.interp(positions_oi, x, y) / window_size * 1000,
                -np.inf,
                0,
                # np.inf,
            )
            deltacor_validation_interpolated[
                window_size_info["ix"], :
            ] = deltacor_interpolated_
            plotdata_oi = plotdata.query("phase in ['test']").query(
                "window_size == @window_size"
            )
            x = plotdata_oi["window_mid"].values.copy()
            y = plotdata_oi["deltacor"].values.copy()
            y[plotdata_oi["qval"] > 0.1] = 0.0
            deltacor_interpolated_ = np.clip(
                np.interp(positions_oi, x, y) / window_size * 1000,
                -np.inf,
                0,
                # np.inf,
            )
            deltacor_test_interpolated[
                window_size_info["ix"], :
            ] = deltacor_interpolated_

            retained_interpolated_ = (
                np.interp(
                    positions_oi, plotdata_oi["window_mid"], plotdata_oi["retained"]
                )
                / window_size
                * 1000
            )
            retained_interpolated[window_size_info["ix"], :] = retained_interpolated_
            lost_interpolated_ = (
                np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["lost"])
                / window_size
                * 1000
            )
            lost_interpolated[window_size_info["ix"], :] = lost_interpolated_

        # save
        interpolated = {
            "deltacor_validation": deltacor_validation_interpolated,
            "deltacor_test": deltacor_test_interpolated,
            "retained": retained_interpolated,
            "lost": lost_interpolated,
        }
        pickle.dump(interpolated, (scores_folder / "interpolated.pkl").open("wb"))
