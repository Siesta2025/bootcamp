import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXent(nn.Module):
    def __init__(self, tau=0.7):
        super().__init__()
        self.tau = tau

    def forward(self, z1, z2):
        assert z1.shape == z2.shape, \
            "Two outputs must have the same shape."

        batch_size = z1.size(0)

        z = torch.cat([z1, z2], dim=0)

        z = F.normalize(z, dim=1)

        logits = z @ z.T
        logits = logits / self.tau

        self_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z.device,
        )
        logits = logits.masked_fill(
            self_mask,
            float("-inf"),
        )


        targets = (
            torch.arange(
                2 * batch_size,
                device=z.device,
            )
            + batch_size
        ) % (2 * batch_size)

        return F.cross_entropy(logits, targets)