import ast
import random
import time
from itertools import chain, combinations

import numpy as np
import torch
import yaml
from decorator import decorator
import wandb


def print_exp_info(args, config, epoch):
    s = (
        f'epoch: [{epoch+1}/{config["num_epochs"]}] {args.device} {args.task_name} '
        f'method: {args.method} K: {args.num_clients} p_miss_train: {args.p_miss_train} '
        f'seed: {args.seed} lr: {config["lr"]} decay: {config["weight_decay"]} '
        f'mom: {config["momentum"]}'
    )
    sep = "-" * len(s)
    print(f"{sep}\n{s}\n{sep}")


def init_wandb(args, config):
    wandb.init(
        project=args.project,
        config={
            "dataset": config["dataset"],
            "architecture": config["model"],
            "cuda": args.cuda_id,
            "method": args.method,
            "lr": config["lr"],
            "n_epochs": config["num_epochs"],
            "seed": args.seed,
            "num_clients": args.num_clients,
            "batch_size": config["batch_size"],
            "weight_decay": config["weight_decay"],
            "momentum": config["momentum"],
        },
        name = (
            args.wandb_name 
            if args.wandb_name 
            else f"{args.task_name}_{args.method}_K{args.num_clients}_p_miss_train{args.p_miss_train}_s{args.seed}"
        ),
    )


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@decorator
def time_decorator(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    print(f"time to run {func.__name__}(): {time.time() - start:.2f}s.")
    return result


def powerset_except_empty(n):
    return list(chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1)))


def setup_task(args):
    from models import get_model
    from optimizers import get_optimizer
    from criterions import get_criterion
    from schedulers import get_scheduler

    with open("configs/task_config.yaml") as file:
        config = yaml.safe_load(file)[args.method][args.task_name][args.num_clients][args.p_miss_train]

    model = get_model(args.method, config["model"], config["dataset"], args, config)
    optimizer = get_optimizer(args.method, config["optimizer"], model, config)
    scheduler = get_scheduler(args.method, config["scheduler"], optimizer, config)
    criterion = get_criterion(config["criterion"])

    method_map = {
        "laser": ("methods.laser", "train_laser", "test_laser"),
        "decoupled": ("methods.decoupled", "train_decoupled", "test_decoupled"),
        "ensemble": ("methods.ensemble", "train_decoupled", "test_ensemble"),
        "plug": ("methods.plug", "train_plug", "test_plug"),
    }
    module_name, train_name, test_name = method_map[args.method]
    mod = __import__(module_name, fromlist=[train_name, test_name])
    train, test = getattr(mod, train_name), getattr(mod, test_name)

    return config, model, optimizer, scheduler, criterion, train, test


def check_sets_of_clients_valid(args):
    if args.blocks_in_tasks_t is not None:
        if args.method == "laser":
            raise ValueError("blocks_in_tasks_t should not be provided when method is 'laser'")
        try:
            args.blocks_in_tasks_t = ast.literal_eval(args.blocks_in_tasks_t)
        except (ValueError, SyntaxError):
            raise ValueError("Invalid format for blocks_in_tasks_t. Must be a valid list of tuples.")
    else:
        args.blocks_in_tasks_t = powerset_except_empty(args.num_clients)


def get_metrics(train_metrics, test_metrics, compute_f1, blocks_in_tasks_t):
    return {
        str(clients): {
            "train_loss": train_metrics["train_loss"][i],
            "train_acc": train_metrics["train_acc"][i],
            "test_loss": test_metrics["test_loss"][i],
            "test_acc": test_metrics["test_acc"][i],
            **(
                {"train_f1": train_metrics["train_f1"][i], "test_f1": test_metrics["test_f1"][i]}
                if compute_f1 else {}
            ),
        }
        for i, clients in enumerate(blocks_in_tasks_t)
    }
