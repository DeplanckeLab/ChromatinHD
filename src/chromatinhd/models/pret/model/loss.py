def paircor(x, y, dim=0, eps=1e-3):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(dim) / divisor
    return cor


def paircor_loss(x, y):
    return -paircor(x, y).mean() * 100


def region_paircor_loss(x, y):
    return -paircor(x, y) * 100


def pairzmse(x, y, dim=0, eps=1e-3):
    y = (y - y.mean(dim, keepdims=True)) / (y.std(dim, keepdims=True) + eps)
    return (y - x).pow(2).mean(dim)


def pairzmse_loss(x, y):
    return pairzmse(x, y).mean() * 0.1


def region_pairzmse_loss(x, y):
    return pairzmse(x, y) * 0.1
