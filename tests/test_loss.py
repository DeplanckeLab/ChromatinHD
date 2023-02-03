import torch
import chromatinhd.loss


class TestLoss:
    def test_loss(self):
        x = torch.stack(
            [
                torch.linspace(0, 10, 100),
                torch.linspace(1, 20, 100),
                torch.linspace(1, 20, 100),
            ],
            dim=0,
        )
        y = x.clone()
        y[0] = y[0] * 5 - 3
        y[1] = y[1] * (-4) + 5

        x2 = chromatinhd.loss.normlinreg(x, y)
        assert (torch.sqrt((x2 - y) ** 2) < 1e-3).all()
