import torch.optim as optim

OPTIMIZERS_DICT = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}

def _create_optimizer(model, name, config):
    try:
        optimizer_class = OPTIMIZERS_DICT[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer name: {name}")

    optimizer_params = {"params": model.parameters(),
        "lr": config["lr"], "weight_decay": config.get("weight_decay", 0),
    }
    
    if name == "sgd":
        optimizer_params["momentum"] = config.get("momentum", 0)

    return optimizer_class(**optimizer_params)

def get_optimizer(method_type: str, name: str, model, config: dict):
    if not isinstance(model, (list, tuple)):
        model = [model]
    
    if method_type in ("laser", "decoupled", "ensemble", "plug"):
        return [_create_optimizer(m, name, config) for m in model]
    else:
        raise ValueError(f"Unknown method type: {method_type}")
