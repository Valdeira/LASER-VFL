import torch.nn as nn

criterions_dict = {
    "cross_entropy": nn.CrossEntropyLoss,
}

def get_criterion(name):
    try:
        return criterions_dict[name]()
    except KeyError:
        raise ValueError(f"Unknown criterion name: {name}")
