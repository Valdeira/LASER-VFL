from models.resnet18 import Resnet18LaserModel
from models.resnet18 import Resnet18DecoupledModel
from models.lstm import DecoupledModel as DecoupledLstm
from models.lstm import LaserModel as LaserLstm
from models.mlp import MLPDecoupledModel
from models.mlp import MLPLaserModel


models_dict = {
                "laser": {
                        "resnet18": Resnet18LaserModel,
                        "lstm": LaserLstm,
                        "mlp": MLPLaserModel,
                        },
                "decoupled":
                        {
                        "resnet18": Resnet18DecoupledModel,
                        "lstm": DecoupledLstm,
                        "mlp": MLPDecoupledModel,
                        },
                "plug":
                        {
                        "resnet18": Resnet18DecoupledModel,
                        "lstm": DecoupledLstm,
                        "mlp": MLPDecoupledModel,
                        },
                "ensemble":
                        {
                        "resnet18": Resnet18DecoupledModel,
                        "lstm": DecoupledLstm,
                        "mlp": MLPDecoupledModel,
                        },
                }


def get_model(method_type, model_name, dataset, args, config):
    try:
        Model = models_dict[method_type][model_name]
        if method_type == "laser":
            if model_name == "lstm":
                from .mimic_model_utils import init as init_mimic
                vocab_d = init_mimic(True, False, False, True, False, False)
                return [Model(dataset, args, vocab_d, config).to(args.device)]
            else:
                return [Model(dataset, args.num_clients).to(args.device)]
        elif method_type in ["decoupled", "ensemble"]:
            if model_name == "lstm":
                from .mimic_model_utils import init as init_mimic
                vocab_d = init_mimic(True, False, False, True, False, False)
                return [Model(dataset, args, vocab_d, config, clients_in_model).to(args.device) for clients_in_model in args.blocks_in_tasks_t]
            else:
                return [Model(dataset, args, clients_in_model).to(args.device) for clients_in_model in args.blocks_in_tasks_t]
        elif method_type == "plug":
            if model_name == "lstm":
                from .mimic_model_utils import init as init_mimic
                vocab_d = init_mimic(True, False, False, True, False, False)
                return [Model(dataset, args, vocab_d, config, clients_in_model, aggregation="conc").to(args.device) for clients_in_model in args.blocks_in_tasks_t]
            else:
                return [Model(dataset, args, clients_in_model, aggregation="conc").to(args.device) for clients_in_model in args.blocks_in_tasks_t]
        else:
            raise ValueError(f"Unknown method type: {method_type}.")
    except KeyError:
        raise ValueError(f"Unknown model name ({model_name}) or method name ({method_type})")
