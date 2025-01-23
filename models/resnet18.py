import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import BasicBlock

from models.model_utils import (FusionModel, BaseLaserModel, BaseDecoupledModel,
                                drop_mask, task_to_hyperparameters)


class Resnet18FeatureExtractor(nn.Module):
    
    def __init__(self, cut_dim, dataset):
        super().__init__()

        self.resnet18 = torchvision.models.resnet18()

        if dataset == 'cifar10':
            # smaller inital kernel_size and remove maxpool due to CIFAR size, instead of ImageNet
            # number of channels reduced in half throughout
            self.resnet18.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=False)
            self.resnet18.bn1 = nn.BatchNorm2d(32)
            self.resnet18.maxpool = nn.Identity()
            self.resnet18.inplanes = 32
            self.resnet18.layer1 = self.resnet18._make_layer(BasicBlock, 32, 2)
            self.resnet18.layer2 = self.resnet18._make_layer(BasicBlock, 64, 2, stride=2)
            self.resnet18.layer3 = self.resnet18._make_layer(BasicBlock, 128, 2, stride=2)
            self.resnet18.layer4 = self.resnet18._make_layer(BasicBlock, 256, 2, stride=2)
            """
            note we do not need to explicitly adjust the network to the input shape when going
            from the default 16x16 to 32x16 due to the default AdaptiveAvgPool2d in resnet 18
            """
            self.resnet18.fc = nn.Linear(256 * BasicBlock.expansion, cut_dim)
        elif dataset == 'cifar100':
            self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet18.maxpool = nn.Identity()
            self.resnet18.fc = nn.Linear(512 * BasicBlock.expansion, cut_dim)
    
    def forward(self, x):
        return self.resnet18(x)


class Resnet18LaserModel(BaseLaserModel):
    
    def __init__(self, dataset, num_clients):
        num_classes, _, cut_dim, _ = task_to_hyperparameters(dataset)
        feature_extractors = nn.ModuleList([Resnet18FeatureExtractor(cut_dim, dataset) for _ in range(num_clients)])
        fusion_models = nn.ModuleList([FusionModel(cut_dim, num_classes) for _ in range(num_clients)])
        self.dataset = dataset
        self.map_idx_to_partition = get_idx_to_partition_map(dataset, num_clients)
        super().__init__(feature_extractors, fusion_models, num_clients)
    
    def get_block(self, x, i):
        [x_] = x
        row_indices, col_indices = self.map_idx_to_partition[i]
        start_row, end_row = row_indices
        start_col, end_col = col_indices
        return x_[:, :, start_row:end_row, start_col:end_col]


class Resnet18DecoupledModel(BaseDecoupledModel):
    def __init__(self, dataset, args, clients_in_model=None, aggregation="mean"): # TODO replace args with num_clients directly
        num_clients = args.num_clients
        self.num_clients = num_clients
        num_classes, _, cut_dim, _ = task_to_hyperparameters(dataset)
        self.map_idx_to_partition = get_idx_to_partition_map(dataset, num_clients)

        # Assign the clients involved in the model (feature extractors used in this head), defaulting to all clients if none specified
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        feature_extractors = nn.ModuleList([Resnet18FeatureExtractor(cut_dim, dataset) for _ in self.clients_in_model])
        fusion_model = FusionModel(cut_dim, num_classes, aggregation, args.num_clients)
        super().__init__(feature_extractors, fusion_model, num_clients)

    def get_block(self, x, i):
        [x_] = x        
        row_indices, col_indices = self.map_idx_to_partition[i]
        start_row, end_row = row_indices
        start_col, end_col = col_indices
        return x_[:, :, start_row:end_row, start_col:end_col]
    

def get_idx_to_partition_map(dataset: str, num_clients: int) -> dict:
    if dataset in ["cifar10", "cifar100"] and num_clients == 10:
        return  {
                0: ((0, 11), (0, 8)),
                1: ((0, 11), (8, 16)),
                2: ((0, 11), (16, 24)),
                3: ((0, 11), (24, 32)),
                4: ((11, 22), (0, 11)),
                5: ((11, 22), (11, 22)),
                6: ((11, 22), (22, 32)),
                7: ((22, 32), (0, 11)),
                8: ((22, 32), (11, 22)),
                9: ((22, 32), (22, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 9:
        return  {
                0: ((0, 11), (0, 11)),
                1: ((0, 11), (11, 22)),
                2: ((0, 11), (22, 32)),
                3: ((11, 22), (0, 11)),
                4: ((11, 22), (11, 22)),
                5: ((11, 22), (22, 32)),
                6: ((22, 32), (0, 11)),
                7: ((22, 32), (11, 22)),
                8: ((22, 32), (22, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 8:
        return  {
                0: ((0, 16), (0, 8)),
                1: ((0, 16), (8, 16)),
                2: ((0, 16), (16, 24)),
                3: ((0, 16), (24, 32)),
                4: ((16, 32), (0, 8)),
                5: ((16, 32), (8, 16)),
                6: ((16, 32), (16, 24)),
                7: ((16, 32), (24, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 7:
        return  {
                0: ((0, 16), (0, 8)),
                1: ((0, 16), (8, 16)),
                2: ((0, 16), (16, 24)),
                3: ((0, 16), (24, 32)),
                4: ((16, 32), (0, 11)),
                5: ((16, 32), (11, 21)),
                6: ((16, 32), (21, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 6:
        return  {
                0: ((0, 16), (0, 11)),
                1: ((0, 16), (11, 21)),
                2: ((0, 16), (21, 32)),
                3: ((16, 32), (0, 11)),
                4: ((16, 32), (11, 21)),
                5: ((16, 32), (21, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 5:
        return  {
                0: ((0, 16), (0, 11)),
                1: ((0, 16), (11, 21)),
                2: ((0, 16), (21, 32)),
                3: ((16, 32), (0, 16)),
                4: ((16, 32), (16, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 4:
        return  {
                0: ((0, 16), (0, 16)),
                1: ((0, 16), (16, 32)),
                2: ((16, 32), (0, 16)),
                3: ((16, 32), (16, 32))
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 3:
        return  {
                0: ((0, 16), (0, 16)),
                1: ((0, 16), (16, 32)),
                2: ((16, 32), (0, 32)),
                }
    elif dataset in ["cifar10", "cifar100"] and num_clients == 2:
        return  {
                0: ((0, 32), (0, 16)),
                1: ((0, 32), (16, 32)),
                }
    else:
        raise NotImplementedError
