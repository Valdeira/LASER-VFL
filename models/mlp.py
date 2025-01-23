import random

import torch
import torch.nn as nn

from utils import powerset_except_empty
from models.model_utils import FusionModel, drop_mask, task_to_hyperparameters


class MLP(nn.Module):
    def __init__(self, input_size, cut_dim, hidden_dim):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cut_dim),
        )

    def forward(self, x):
        return self.hidden_layers(x)


class LaserModel(nn.Module):
    
    def __init__(self, dataset, num_clients):
        super().__init__()
        self.num_clients = num_clients
        self.powerset = powerset_except_empty(self.num_clients)
        num_classes, input_size, cut_dim, hidden_dim = task_to_hyperparameters(dataset)

        self.feature_extractors = nn.ModuleList([MLP(input_size, cut_dim, hidden_dim) for _ in range(self.num_clients)])
        self.fusion_models = nn.ModuleList([FusionModel(cut_dim, num_classes) for _ in range(self.num_clients)])

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
    
    def get_block(self, x, index):
        [x_] = x
        block_size = x_.shape[1] // self.num_clients
        start = index * block_size
        end = start + block_size
        return x_[:, start:end]


class DecoupledModel(nn.Module):

    def __init__(self, dataset, args, clients_in_model=None, aggregation="mean"):
        super().__init__()
        self.num_clients = args.num_clients
        num_classes, input_size, cut_dim, hidden_dim = task_to_hyperparameters(dataset)

        # Assign the clients involved in the model (feature extractors used in this head), defaulting to all clients if none specified
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        self.feature_extractors = nn.ModuleList([MLP(input_size, cut_dim, hidden_dim) for _ in self.clients_in_model])
        self.fusion_model = FusionModel(cut_dim, num_classes, aggregation, args.num_clients)

    def get_block(self, x, index):
        [x_] = x
        block_size = x_.shape[1] // self.num_clients
        start = index * block_size
        end = start + block_size
        return x_[:, start:end]
    
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
