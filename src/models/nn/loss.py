import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, n_views, temperature):
        super().__init__()
        self.n_views = n_views
        self.temperature = temperature

    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor):
        device = feats1.device
        features = torch.cat([feats1, feats2])

        labels = torch.cat([torch.arange(feats1.shape[0])] * self.n_views, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).type_as(feats1)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        logits = logits / self.temperature
        return logits, labels
