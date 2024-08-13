import torch

DEFAULT_MIN_BIN_HEIGHT = 1e-5


def calculate_heights(unnormalized_heights, widths, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, local=True):
    unnorm_heights_exp = torch.exp(unnormalized_heights)

    min_bin_height = 1e-10

    if local:
        # per feature normalization
        unnormalized_area = torch.sum(
            ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths,
            dim=-1,
            keepdim=True,
        )
        heights = unnorm_heights_exp / unnormalized_area
        heights = min_bin_height + (1 - min_bin_height) * heights
    else:
        # global normalization
        unnormalized_area = torch.sum(
            ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths,
        )
        heights = unnorm_heights_exp * unnorm_heights_exp.shape[-2] / unnormalized_area
        heights = min_bin_height + (1 - min_bin_height) * heights

    # to check
    # normalized_area = torch.sum(
    #     ((heights[..., :-1] + heights[..., 1:]) / 2) * widths,
    #     dim=-1,
    #     keepdim=True,
    # )
    # print(normalized_area.sum())

    return heights


def calculate_bin_left_cdf(heights, widths):
    bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
    bin_left_cdf[..., -1] = 1.0
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode="constant", value=0.0)
    return bin_left_cdf
