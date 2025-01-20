import torch.optim as optim

optimizers_dict = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}

def create_optimizer(optimizer_class, model, config, name):
    """Helper function to create an optimizer with given parameters."""
    optimizer_params = {
        "params": model.parameters(),
        "lr": config["lr"],
        "weight_decay": config.get("weight_decay", 0)
    }
    if name == "sgd":
        optimizer_params["momentum"] = config.get("momentum", 0)
    return optimizer_class(**optimizer_params)

def get_optimizer(method: str, name: str, model, config: dict):
    """
    Select and configure optimizer based on the method and name.

    Parameters:
    - method (str): The optimization method ("moo" or "decoupled").
    - name (str): The name of the optimizer ("adam" or "sgd").
    - model: The model or list of models to optimize.
    - config (dict): Configuration dictionary containing "lr" and optional "weight_decay" and "momentum".

    Returns:
    - Optimizer or list of optimizers.
    """
    try:
        optimizer_class = optimizers_dict[name]
    except KeyError:
        raise ValueError(f"Unknown optimizer name: {name}")

    if method == "moo":
        return [create_optimizer(optimizer_class, m, config, name) for m in model]
    elif method == "decoupled" or "ensemble":
        return [create_optimizer(optimizer_class, m, config, name) for m in model]
    else:
        raise ValueError(f"Unknown method: {method}")
