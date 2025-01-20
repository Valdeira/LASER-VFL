import math
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, embedding_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(f'x.reshape(x.shape[0], -1).shape', x.reshape(x.shape[0], -1).shape)
        print(f'self.fc.weight.shape', self.fc.weight.shape)
        exit()

        x = self.fc(x.reshape(x.shape[0], -1))
        return self.sigmoid(x)

class FusionModel(nn.Module):
    def __init__(self, aggregation, embedding_dim, num_classes):
        super().__init__()
        if aggregation not in ['sum', 'mean']:
            raise ValueError('Invalid aggregation mechanism.')
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.aggregation = aggregation

    def forward(self, x):
        if self.aggregation == 'sum':
            x = torch.stack(x).sum(dim=0)
        elif self.aggregation == 'mean':
            x = torch.stack(x).mean(dim=0)
        x = self.fc(x)
        return x

class VFLModel(nn.Module):
    def __init__(self, local_data_dim, cut_dim, num_classes, args, clients_in_model=None, shared_feature_extractors=None, pixels_per_axis=28):
        super().__init__()
        
        # Assign the clients involved in the model, defaulting to all clients if none specified
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        # Initialize feature extractors either as shared or individual per client
        if shared_feature_extractors is None:
            # If no shared extractors provided, create an individual extractor for each client
            self.feature_extractors = nn.ModuleList([FeatureExtractor(local_data_dim, cut_dim) for _ in self.clients_in_model])
        else:
            # Use provided shared feature extractors ensuring they match the client indices
            self.feature_extractors = nn.ModuleList([shared_feature_extractors[i] for i in self.clients_in_model])

        # Initialize the fusion model to aggregate outputs from feature extractors
        self.fusion_model = FusionModel(args.aggregation, cut_dim, num_classes)

        self.blocks_per_axis = int(math.sqrt(args.num_clients))
        if pixels_per_axis % self.blocks_per_axis != 0:
            raise ValueError('Pixels per axis not divisible by blocks per axis')
        self.pixels_per_block = pixels_per_axis // self.blocks_per_axis

    def forward(self, x):
        embeddings = []
        for i, j in enumerate(self.clients_in_model):
            local_input = x[:, :, j // self.blocks_per_axis * self.pixels_per_block : (j // self.blocks_per_axis + 1) * self.pixels_per_block,
                                  j % self.blocks_per_axis * self.pixels_per_block : (j % self.blocks_per_axis + 1) * self.pixels_per_block]
            embeddings.append(self.feature_extractors[i](local_input))
        return self.fusion_model(embeddings)
