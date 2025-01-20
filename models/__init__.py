from .resnet18 import LaserModel as LaserResnet
from .resnet18 import DecoupledModel as DecoupledResnet
from .lstm import DecoupledModel as DecoupledLstm
from .lstm import LaserModel as LaserLstm
from .mlp import DecoupledModel as DecoupledMLP
from .mlp import LaserModel as LaserMLP
from .mlp_small import DecoupledModel as DecoupledMLPSmall
from .mlp_small import LaserModel as LaserMLPSmall

models_dict = {
                "laser": {
                        "resnet18": LaserResnet,
                        "lstm": LaserLstm,
                        "mlp": LaserMLP,
                        "mlp_small": LaserMLPSmall,
                        },
                "decoupled":
                        {
                        "resnet18": DecoupledResnet,
                        "lstm": DecoupledLstm,
                        "mlp": DecoupledMLP,
                        "mlp_small": DecoupledMLPSmall,
                        },
                "plug":
                        {
                        "resnet18": DecoupledResnet,
                        "lstm": DecoupledLstm,
                        "mlp": DecoupledMLP,
                        "mlp_small": DecoupledMLPSmall,
                        },
                "ensemble":
                        {
                        "resnet18": DecoupledResnet,
                        "lstm": DecoupledLstm,
                        "mlp": DecoupledMLP,
                        "mlp_small": DecoupledMLPSmall,
                        },
                }


def get_model(method, model_name, dataset, args, config):
    try:
        Model = models_dict[method][model_name]
        if method == "laser":
            if model_name == "lstm":
                from .mimic_model_utils import init as init_mimic
                vocab_d = init_mimic(True, False, False, True, False, False)
                return [Model(dataset, args, vocab_d, config).to(args.device)]
            else:
                return [Model(dataset, args.num_clients).to(args.device)]
        elif method in ["decoupled", "ensemble"]:
            if method == "ensemble":
                assert args.blocks_in_tasks_t == [(i,) for i in range(args.num_clients)]
            if model_name == "lstm":
                from .mimic_model_utils import init as init_mimic
                vocab_d = init_mimic(True, False, False, True, False, False)
                return [Model(dataset, args, vocab_d, config, clients_in_model).to(args.device) for clients_in_model in args.blocks_in_tasks_t]
            else:
                return [Model(dataset, args, clients_in_model).to(args.device) for clients_in_model in args.blocks_in_tasks_t]
        elif method == "plug":
            if model_name == "lstm":
                from .mimic_model_utils import init as init_mimic
                vocab_d = init_mimic(True, False, False, True, False, False)
                return [Model(dataset, args, vocab_d, config, clients_in_model, aggregation="conc").to(args.device) for clients_in_model in args.blocks_in_tasks_t]
            else:
                return [Model(dataset, args, clients_in_model, aggregation="conc").to(args.device) for clients_in_model in args.blocks_in_tasks_t]
        else:
            raise ValueError(f"Unknown method: {method}.")
    except KeyError:
        raise ValueError(f"Unknown model name ({model_name}) or method name ({method})")
