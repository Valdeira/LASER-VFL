# if --method moo: implementation of our robust VFL method
# if --method decoupled: implementation of up to a powerset of VFL models based on the subsets of blocks

import argparse
import wandb
import torch

from utils import (time_decorator, print_exp_info, init_wandb, setup_task,
                    set_seed, get_dataloaders, handle_sets_of_clients_in_tasks, get_metrics)


@time_decorator
def main(args):
    
    set_seed(args.seed)
    config, models, optimizers, schedulers, criterion, train, test = setup_task(args)
    train_loader, test_loader = get_dataloaders(config["dataset"], config["batch_size"], args, num_workers=config["num_workers"])

    if args.use_wandb:
        init_wandb(args, config)
    
    print('Computing initial metrics...')
    compute_f1 = True if args.task_name in ["mimic4", "credit"] else False
    train_metrics = test(train_loader, models, criterion, args.device, compute_f1=compute_f1, is_train_data=True)
    test_metrics = test(test_loader, models, criterion, args.device, compute_f1=compute_f1)
    if args.use_wandb:
        wandb.log(get_metrics(train_metrics, test_metrics, compute_f1, args.sets_of_clients_in_tasks))
    
    for epoch in range(config["num_epochs"]):
        print_exp_info(args, config, epoch)
        train_metrics = train(train_loader, models, optimizers, criterion, args, compute_f1=compute_f1)
        test_metrics = test(test_loader, models, criterion, args.device, compute_f1=compute_f1)
        for scheduler in schedulers:
            scheduler.step()
        if args.use_wandb:
            wandb.log(get_metrics(train_metrics, test_metrics, compute_f1, args.sets_of_clients_in_tasks))

    final_p_miss_test_l = [0.0, 0.1, 0.5, None]
    for final_p_miss_test in final_p_miss_test_l:
        _, test_loader = get_dataloaders(config["dataset"], config["batch_size"], args, num_workers=config["num_workers"], final_p_miss_test=final_p_miss_test)
        test_metrics = test(test_loader, models, criterion, args.device, compute_f1=compute_f1, is_final=True)
        if args.use_wandb:
            metrics = {f'final_test_acc_{final_p_miss_test}': test_metrics["final_test_acc"], **({f'final_test_f1_{final_p_miss_test}': test_metrics["final_test_f1"]} if compute_f1 else {})}
            wandb.log(metrics)
        print_str = f'(final_p_miss_test {final_p_miss_test}) final_test_acc: {test_metrics["final_test_acc"]}'
        print_str += (f' | final_test_f1: {test_metrics["final_test_f1"]}' if compute_f1 else '')
        print(print_str)
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='cifar10') # hapt|credit|mimic4|cifar10|cifar100
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--wandb_name', help='Name of the run.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--p_miss_train', type=float)
    parser.add_argument('--p_miss_test', type=float, default=0.0)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb', help='Disable wandb logging.')
    parser.add_argument('--l2_loss', action='store_true')
    parser.add_argument('--method', choices=['decoupled', 'moo', 'ensemble', 'plug'], required=True)
    parser.add_argument('--sets_of_clients_in_tasks', type=str, help='List of tuples representing sets of clients in tasks')
    parser.add_argument('--p_drop', type=float, default=0.0)
    args = parser.parse_args()

    args.project = 'vfl-sandbox'
    args.device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')

    handle_sets_of_clients_in_tasks(args)

    main(args)

# NOTE: mimic4 code assumes that data is ICU, that features are diagnosis and chart, and that the model is a 'Time-series LSTM'
# NOTE: hapt experiment uses
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
# criterion = nn.CrossEntropyLoss()
# no scheduler
# TODO set wandb_name from other args (may use only if None)
# TODO remove --p_miss_test argument (we're not using it)
