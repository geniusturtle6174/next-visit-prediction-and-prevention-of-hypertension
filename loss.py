import torch


def mse_loss(pred, target):
    return torch.mean((target / 100.0) * (pred - target) ** 2)


def f_loss(pred, target, beta=1):
    tp = torch.sum(pred * target, dim=1)
    fp = torch.sum(pred * (1 - target), dim=1)
    fn = torch.sum((1 - pred) * target, dim=1)
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-6)
    f = torch.nan_to_num(f)
    return 1 - f.mean()
