def paircor(x, y, dim=0, eps=0.1):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor


def paircor_loss(x, y):
    return -paircor(x, y).mean() * 100


def gene_paircor_loss(x, y):
    return -paircor(x, y) * 100
