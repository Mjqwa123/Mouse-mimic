"""GRU + mixture density network that predicts the next mouse step as a distribution."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_FEATURES


class MimicNet(nn.Module):
    def __init__(self, in_dim=N_FEATURES, hidden=128, layers=2, mixtures=8, dropout=0.1):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden=hidden, layers=layers, mixtures=mixtures, dropout=dropout)
        self.k = mixtures
        self.gru = nn.GRU(in_dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        # per mixture: weight, mean (2), log sigma (2), correlation; plus one end-of-move logit
        self.head = nn.Linear(hidden, mixtures * 6 + 1)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        p = self.head(out)
        k = self.k
        logit_pi = p[..., :k]
        mu = p[..., k:3 * k].unflatten(-1, (k, 2))
        log_sigma = p[..., 3 * k:5 * k].unflatten(-1, (k, 2)).clamp(-5.0, 3.0)
        rho = torch.tanh(p[..., 5 * k:6 * k]) * 0.99
        end_logit = p[..., 6 * k]
        return (logit_pi, mu, log_sigma, rho, end_logit), h


def mdn_log_prob(params, target):
    """Log-likelihood of target (..., 2) under the bivariate Gaussian mixture."""
    logit_pi, mu, log_sigma, rho, _ = params
    z = (target.unsqueeze(-2) - mu) / log_sigma.exp()
    zx, zy = z[..., 0], z[..., 1]
    one_minus = 1.0 - rho ** 2
    q = (zx ** 2 + zy ** 2 - 2.0 * rho * zx * zy) / one_minus
    log_n = -math.log(2 * math.pi) - log_sigma.sum(-1) - 0.5 * torch.log(one_minus) - 0.5 * q
    return torch.logsumexp(F.log_softmax(logit_pi, dim=-1) + log_n, dim=-1)


def loss_fn(params, target, end, mask):
    """Masked mean mixture NLL and end-of-move BCE."""
    n = mask.sum()
    nll = -(mdn_log_prob(params, target) * mask).sum() / n
    bce = (F.binary_cross_entropy_with_logits(params[4], end, reduction="none") * mask).sum() / n
    return nll, bce


def sample_step(params, temperature, rng):
    """Sample one (dx, dy) and the end probability from single-step params."""
    logit_pi, mu, log_sigma, rho, end_logit = (p.reshape(p.shape[2:]).numpy().astype(float) for p in params)
    logits = logit_pi / max(temperature, 1e-3)
    pi = np.exp(logits - logits.max())
    pi /= pi.sum()
    k = rng.choice(len(pi), p=pi)
    sx, sy = np.exp(log_sigma[k]) * math.sqrt(temperature)
    r = rho[k]
    z1, z2 = rng.standard_normal(2)
    dx = mu[k, 0] + sx * z1
    dy = mu[k, 1] + sy * (r * z1 + math.sqrt(1.0 - r * r) * z2)
    p_end = 1.0 / (1.0 + math.exp(-float(end_logit)))
    return np.array([dx, dy]), p_end
