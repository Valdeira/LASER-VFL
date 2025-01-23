import random

import torch
import torch.nn as nn

from utils import powerset_except_empty

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


class BaseLaserModel(nn.Module):
    
    def __init__(self, feature_extractors, fusion_models, num_clients):
        super().__init__()
        self.num_clients = num_clients
        self.powerset = powerset_except_empty(num_clients)
        self.feature_extractors = feature_extractors
        self.fusion_models = fusion_models

    def forward(self, x, training=True, observed_blocks=None):

        if observed_blocks is None: # NOTE this is be the case e.g. for test data
            observed_blocks = list(range(self.num_clients))
        
        embeddings = {}
        for i in observed_blocks:
            local_input = self.get_block(x, i)
            embeddings[i] = self.feature_extractors[i](local_input)

        if training:
            outputs = []
            for i, fusion_model in enumerate(self.fusion_models):
                if i not in observed_blocks:
                    continue
                sets_considered_by_head = [clients_l for clients_l in self.powerset if i in clients_l and set(clients_l).issubset(set(observed_blocks))]
                head_output = {}
                for num_clients_in_agg in range(1, len(observed_blocks) + 1):
                    set_to_sample = [client_set for client_set in sets_considered_by_head if len(client_set) == num_clients_in_agg]
                    [sample] = random.sample(set_to_sample, 1)
                    head_output[sample] = fusion_model([embeddings[j] for j in sample])
                outputs.append(head_output)
        else:
            outputs = [{clients_l: fusion_model([embeddings[j] for j in clients_l]) for clients_l in self.powerset if i in clients_l} for i, fusion_model in enumerate(self.fusion_models)]

        return outputs
    
    def get_block(self, x, i):
        raise NotImplementedError("Subclasses must override this method.")


class BaseDecoupledModel(nn.Module):
    def __init__(self, feature_extractors, fusion_model, num_clients):
        super().__init__()
        self.num_clients = num_clients
        self.feature_extractors = feature_extractors
        self.fusion_model = fusion_model

    def forward(self, x, plug_mask=None, p_drop=0):
        
        if plug_mask is not None: # for PlugVFL we use mask in the forward pass
            """
            missing feature blocks are always zeros at the fusion model;
            during training, p_drop>0 is provided and the observed feature blocks
            can also  be dropped, leading to zeros at the fusion model w.p. p_drop in (0,0.5)
            """
            new_mask = drop_mask(plug_mask, p_drop)
            embeddings = [
                self.feature_extractors[i](self.get_block(x, j)) if new_mask[i] else 
                torch.zeros_like(self._get_dummy_output(self.feature_extractors[i], self.get_block(x, j)))
                for i, j in enumerate(self.clients_in_model)
            ]
            return self.fusion_model(embeddings)
        else:
            embeddings = [self.feature_extractors[i](self.get_block(x, j)) for i, j in enumerate(self.clients_in_model)]
            return self.fusion_model(embeddings)
    
    def _get_dummy_output(self, feature_extractor, input_tensor):
        with torch.no_grad():
            return feature_extractor(input_tensor)

    def get_block(self, x, i):
        raise NotImplementedError("Subclasses must override this method.")

        
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
