import argparse
import wandb
import torch

from utils import (time_decorator, print_exp_info, init_wandb, setup_task,
                    set_seed, get_dataloaders, check_sets_of_clients_valid,
                    get_metrics)


@time_decorator
def main(args: argparse.Namespace) -> None:
    
    set_seed(args.seed)
    check_sets_of_clients_valid(args)
    config, models, optimizers, schedulers, criterion, train, test = setup_task(args)
    train_loader, test_loader = get_dataloaders(args, config)
    if args.use_wandb:
        init_wandb(args, config)
    
    # compute initial metrics
    compute_f1 = True if args.task_name in ["mimic4", "credit"] else False
    train_metrics = test(train_loader, models, criterion, args.device, compute_f1=compute_f1, is_train_data=True)
    test_metrics = test(test_loader, models, criterion, args.device, compute_f1=compute_f1)
    if args.use_wandb:
        wandb.log(get_metrics(train_metrics, test_metrics, compute_f1, args.blocks_in_tasks_t))
    
    # train
    for epoch in range(config["num_epochs"]):
        print_exp_info(args, config, epoch)
        train_metrics = train(train_loader, models, optimizers, criterion, args, compute_f1=compute_f1)
        test_metrics = test(test_loader, models, criterion, args.device, compute_f1=compute_f1)
        for scheduler in schedulers:
            scheduler.step()
        if args.use_wandb:
            wandb.log(get_metrics(train_metrics, test_metrics, compute_f1, args.blocks_in_tasks_t))

    # test
    for p_miss_test in args.final_p_miss_test_l:
        _, test_loader = get_dataloaders(args, config, p_miss_test)
        test_metrics = test(test_loader, models, criterion, args.device, compute_f1=compute_f1, is_final=True)
        metrics = {f'final_test_acc_{p_miss_test}': test_metrics["final_test_acc"]}
        print_str = (f"(p_miss_test {p_miss_test}) "
                     f"final_test_acc: {test_metrics['final_test_acc']}")
        if compute_f1:
            metrics[f'final_test_f1_{p_miss_test}'] = test_metrics["final_test_f1"]
            print_str += f" | final_test_f1: {test_metrics['final_test_f1']}"
        if args.use_wandb:
            wandb.log(metrics)
        print(print_str)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', choices=['hapt', 'credit', 'mimic4', 'cifar10', 'cifar100'])
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--wandb_name', help='Name of the run.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--p_miss_train', type=float, default=0.0)
    parser.add_argument('--final_p_miss_test_l', type=lambda x: None if x == "None" else float(x), nargs='*',
                        default=[0.0, 0.1, 0.5, None], help="List of missing probabilities for testing.")
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb', help='Disable wandb logging.')
    parser.add_argument('--method', choices=['decoupled', 'ensemble', 'plug', 'laser'], required=True)
    parser.add_argument('--blocks_in_tasks_t', type=str, help='Tuple of sets of blocks/clients in tasks.')
    parser.add_argument('--p_drop', type=float, default=0.0)
    args = parser.parse_args()

    args.project = 'vfl-sandbox'
    args.device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')

    main(args)
