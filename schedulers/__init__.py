import torch.optim as optim

schedulers_dict = {
    "cosine_annealing_lr": optim.lr_scheduler.CosineAnnealingLR
}

def create_scheduler(scheduler_class, optimizer, config, name):
    """Helper function to create a scheduler with given parameters."""
    if name == "cosine_annealing_lr":
        eta_min_ratio = config.get("eta_min_ratio", 0.1)  # Assuming a default ratio if not specified
        return scheduler_class(optimizer, T_max=config["num_epochs"], eta_min=config["lr"] * eta_min_ratio)

def get_scheduler(method: str, name: str, optimizer, config: dict):
    """
    Select and configure scheduler based on the method and name.

    Parameters:
    - method (str): The scheduling method ("moo" or "decoupled").
    - name (str): The name of the scheduler ("cosine_annealing_lr").
    - optimizer: The optimizer or list of optimizers to apply the scheduler to.
    - config (dict): Configuration dictionary containing "num_epochs", "lr", and optional "eta_min_ratio".

    Returns:
    - Scheduler or list of schedulers.
    """
    if name == "n/a":
        return []
        
    try:
        scheduler_class = schedulers_dict[name]
    except KeyError:
        raise ValueError(f"Unknown scheduler name: {name}")

    if method == "moo":
        return [create_scheduler(scheduler_class, opt, config, name) for opt in optimizer]
    elif method == "decoupled" or "ensemble":
        return [create_scheduler(scheduler_class, opt, config, name) for opt in optimizer]
    else:
        raise ValueError(f"Unknown method: {method}")
