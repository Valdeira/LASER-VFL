import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import BasicBlock

from models.model_utils import BaseLaserModel, BaseDecoupledModel, FusionModel, task_to_hyperparameters
from data.cifar_partitions import CIFAR_PARTITIONS


class Resnet18FeatureExtractor(nn.Module):
    def __init__(self, cut_dim, dataset):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18()

        if dataset == 'cifar10':
            # Modify layers for CIFAR-10
            first_conv_out = 32
            final_inplanes = 256 * BasicBlock.expansion
            self.resnet18.conv1 = nn.Conv2d(3, first_conv_out, kernel_size=3, stride=1, padding=0, bias=False)
            self.resnet18.bn1 = nn.BatchNorm2d(first_conv_out)
            self.resnet18.maxpool = nn.Identity()
            self.resnet18.inplanes = first_conv_out
            self.resnet18.layer1 = self.resnet18._make_layer(BasicBlock, 32, 2)
            self.resnet18.layer2 = self.resnet18._make_layer(BasicBlock, 64, 2, stride=2)
            self.resnet18.layer3 = self.resnet18._make_layer(BasicBlock, 128, 2, stride=2)
            self.resnet18.layer4 = self.resnet18._make_layer(BasicBlock, 256, 2, stride=2)
        elif dataset == 'cifar100':
            # Modify layers for CIFAR-100
            first_conv_out = 64
            final_inplanes = 512 * BasicBlock.expansion
            self.resnet18.conv1 = nn.Conv2d(3, first_conv_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet18.maxpool = nn.Identity()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        self.resnet18.fc = nn.Linear(final_inplanes, cut_dim)
    
    def forward(self, x):
        return self.resnet18(x)


class Resnet18LaserModel(BaseLaserModel):
    
    def __init__(self, dataset, num_clients):
        num_classes, _, cut_dim, _ = task_to_hyperparameters(dataset)
        feature_extractors = nn.ModuleList([Resnet18FeatureExtractor(cut_dim, dataset) for _ in range(num_clients)])
        fusion_models = nn.ModuleList([FusionModel(cut_dim, num_classes) for _ in range(num_clients)])
        self.map_idx_to_partition = get_idx_to_partition_map(dataset, num_clients)
        super().__init__(feature_extractors, fusion_models, num_clients)
    
    def get_block(self, x, i):
        return slice_cifar_block(x[0], self.map_idx_to_partition[i])


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
        return slice_cifar_block(x[0], self.map_idx_to_partition[i])
    

def get_idx_to_partition_map(dataset: str, num_clients: int) -> dict:
    if dataset not in ("cifar10", "cifar100"):
        raise NotImplementedError(f"Only 'cifar10' and 'cifar100' are supported, got {dataset}.")    
    if num_clients not in CIFAR_PARTITIONS:
        raise NotImplementedError(f"No partition map for num_clients={num_clients} in {dataset}.")
    return CIFAR_PARTITIONS[num_clients]


def slice_cifar_block(x, partition):
    (row_start, row_end), (col_start, col_end) = partition
    return x[:, :, row_start:row_end, col_start:col_end]
