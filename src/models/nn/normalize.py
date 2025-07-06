import torch


def normalize(x, order=2, axis=1):
    norm = torch.norm(x, order, axis, True)
    output = torch.div(x, norm)
    return output
