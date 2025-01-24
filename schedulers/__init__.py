import torch.optim as optim

SCHEDULERS_DICT = {
    "cosine_annealing_lr": optim.lr_scheduler.CosineAnnealingLR,
}


def _create_scheduler(scheduler_name: str, optimizer, config: dict):
    scheduler_class = SCHEDULERS_DICT.get(scheduler_name)
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")

    if scheduler_name == "cosine_annealing_lr":
        eta_min_ratio = config.get("eta_min_ratio", 0.1)
        return scheduler_class(optimizer,
            T_max=config["num_epochs"], eta_min=config["lr"] * eta_min_ratio)
    raise ValueError(f"Scheduler not implemented: {scheduler_name}")


def get_scheduler(method_type: str, scheduler_name: str, optimizers, config: dict):
    if scheduler_name == "n/a":
        return []

    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    if method_type in ("laser", "decoupled", "ensemble"):
        return [_create_scheduler(scheduler_name, opt, config) for opt in optimizers]
    else:
        raise ValueError(f"Unknown method type: {method_type}")
