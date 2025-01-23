import torch
import torch.nn as nn


class FusionModel(nn.Module):

    def __init__(self, cut_dim, num_classes, aggregation="mean", num_clients=None):
        super().__init__()
        self.aggregation = aggregation
        if aggregation == 'conc':
            assert num_clients is not None
        fusion_input_dim = cut_dim * num_clients if aggregation == "conc" else cut_dim
        self.classifier = nn.Linear(fusion_input_dim, num_classes)

    def forward(self, x):
        if self.aggregation == 'sum':
            x = torch.stack(x).sum(dim=0)
        elif self.aggregation == 'mean':
            x = torch.stack(x).mean(dim=0)
        elif self.aggregation == 'conc':
            x = torch.cat(x, dim=1)
        pooled_view = self.classifier(x)
        return pooled_view

        
def drop_mask(plug_mask: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Return a new mask, dropping elements with probability p_drop.
    The last element is never dropped."""
    keep = torch.rand_like(plug_mask, dtype=torch.float32) >= p_drop
    keep[-1] = True  # ensure the last element is kept
    return plug_mask & keep


def task_to_hyperparameters(dataset):

    num_classes, input_size, cut_dim, hidden_dim = None, None, None, None

    if dataset == "hapt":
        num_classes = 12
        input_size = 140 # 560 / 4
        cut_dim = 128
        hidden_dim = 128
    elif dataset == "credit":
        num_classes = 2
        input_size = 5 # 20 // 4
        cut_dim = 10
        hidden_dim = 10
    elif dataset == 'cifar10':
        num_classes = 10
        cut_dim = 1024
    elif dataset == 'cifar100':
        num_classes = 100
        cut_dim = 1024
    else:
        raise ValueError(f"Unexpected dataset {dataset}")

    return num_classes, input_size, cut_dim, hidden_dim
