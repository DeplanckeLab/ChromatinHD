import torch
import numpy as np

def score_fragments(
    genemapping,
    coordinates,
    positiongene_scores,
    window,
    cutwindow,
    window_width = None,
    cutwindow_width = None
):
    """
    Scores fragments, mapped to gene regions, by pooling scores close to the cut sites

    :param genemapping 
    """
    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 2
    assert genemapping.shape[0] == coordinates.shape[0]

    if window_width is None:
        window_width = window[1] - window[0]
    else:
        assert window[1] - window[0] == window_width
    if cutwindow_width is None:
        cutwindow_width = cutwindow[1] - cutwindow[0]
    else:
        assert cutwindow[1] - cutwindow[0] == cutwindow_width
    assert (positiongene_scores.shape[0] % window_width) == 0

    # positiongene_scores has dimension [positions x genes, channels(=motifs)]
    
    # has dimensions [fragments, cut sites(0, 1), positions(cutwindow_width)]
    # +1 is added here to center the cut site around cutwindow[0] and cutwindow[1]+1
    idxs = coordinates[:, :, None] - window[0] + torch.arange(cutwindow[0], cutwindow[1]+1, device = coordinates.device)[None, None, :]
    idxs = torch.clamp(idxs, 0, window_width-1)

    # flatten index along different genes using genemapping
    unwrapped_idx = (idxs + genemapping[:, None, None] * window_width)
    
    import time

    start = time.time()
    # extract the relevant scores, has dimensions [positions x fragments, channels(=motifs)]
    view = positiongene_scores[unwrapped_idx.flatten()]
    print(time.time() - start)

    
    # start = time.time()
    # pool for each fragment
    pom = view.reshape((*unwrapped_idx.shape, positiongene_scores.shape[-1])).sum((1, 2))
    # print(pom.shape)
    # fragment_scores = (torch.nn.functional.avg_pool1d(view.transpose(1, 0), cutwindow_width+1)).transpose(1, 0)
    # print(time.time() - start)

    # pool over cut sites 0 and 1
    fragment_scores = pom
    # fragment_scores = fragment_scores.reshape((fragment_scores.shape[0]//2, 2, fragment_scores.shape[-1])).sum(1) #!
    # fragment_scores = fragment_scores.reshape((fragment_scores.shape[0]//2, 2, fragment_scores.shape[-1]))[:, 0]
    
    return fragment_scores

    