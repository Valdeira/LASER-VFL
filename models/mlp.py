import torch
import torch.nn as nn

from models.model_utils import (FusionModel, BaseLaserModel, BaseDecoupledModel,
                                drop_mask, task_to_hyperparameters)


class MLPFeatureExtractor(nn.Module):
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


class MLPLaserModel(BaseLaserModel):
    
    def __init__(self, dataset, num_clients):
        num_classes, input_size, cut_dim, hidden_dim = task_to_hyperparameters(dataset)
        feature_extractors = nn.ModuleList([MLPFeatureExtractor(input_size, cut_dim, hidden_dim) for _ in range(num_clients)])
        fusion_models = nn.ModuleList([FusionModel(cut_dim, num_classes) for _ in range(num_clients)])
        super().__init__(feature_extractors, fusion_models, num_clients)
    
    def get_block(self, x, i):
        [x_] = x
        block_size = x_.shape[1] // self.num_clients
        start = i * block_size
        end = start + block_size
        return x_[:, start:end]


class MLPDecoupledModel(BaseDecoupledModel):
    def __init__(self, dataset, args, clients_in_model=None, aggregation="mean"): # TODO replace args with num_clients directly
        num_clients = args.num_clients
        self.num_clients = num_clients
        num_classes, input_size, cut_dim, hidden_dim = task_to_hyperparameters(dataset)

        # Assign the clients involved in the model (feature extractors used in this head), defaulting to all clients if none specified
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        feature_extractors = nn.ModuleList([MLPFeatureExtractor(input_size, cut_dim, hidden_dim) for _ in self.clients_in_model])
        fusion_model = FusionModel(cut_dim, num_classes, aggregation, args.num_clients)
        super().__init__(feature_extractors, fusion_model, num_clients)

    def get_block(self, x, i):
        [x_] = x
        block_size = x_.shape[1] // self.num_clients
        start = i * block_size
        end = start + block_size
        return x_[:, start:end]
    