import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from utils import powerset_except_empty


def get_final_test_accs(run_name, runs, run_metrics_groups):
    
    for run in runs:
        if run.name == run_name:
            break
    else:
        print("No run named {run_name} found.")

    run_group_dfs = []
    for run_metrics_group in run_metrics_groups:
        # print(f'run_metrics_group: {run_metrics_group}')
        data = [row for row in run.scan_history(keys=run_metrics_group)]
        # print(f'data: {data}')
        run_group_dfs.append(pd.DataFrame(data))

    # print(f'run_group_dfs: {run_group_dfs}')
    # exit()

    run_group_test_acc_dfs = [df.filter(like='test_acc') for df in run_group_dfs]

    # print(f'run_group_test_acc_dfs: {run_group_test_acc_dfs}')
    # print(f'type(run_group_test_acc_dfs): {type(run_group_test_acc_dfs)}')
    # print(f'len(run_group_test_acc_dfs): {len(run_group_test_acc_dfs)}')
    
    # TODO make index in line below an argument
    run_group_test_acc_df = run_group_test_acc_dfs[0] # i=0,1,2,3 for metric of predictor using subset of i+1 clients

    # print(f'run_group_test_acc_df: {run_group_test_acc_df}')
    
    final_test_accs_l = run_group_test_acc_df.iloc[-1].tolist()

    print(f'final_test_accs_l: {final_test_accs_l}')
    # exit()
    
    return final_test_accs_l # list bcs if using e.g. run_group_test_acc_dfs[-2] instead, we'll have accs for [(1,2,3), (1,2,4),...]

def main(runs_names, wandb_project, n_clients):
    
    run_metrics_groups = [[], [], [], []] # [sets with one client, ..., sets with n_clients clients]
    for run_prefix in powerset_except_empty(n_clients):
        base_name = str(run_prefix)
        run_metrics_groups[len(run_prefix) - 1].extend([base_name + '.train_loss', base_name + '.train_acc', base_name + '.test_loss', base_name + '.test_acc'])
    
    api = wandb.Api()
    runs = api.runs(path=wandb_project)

    final_test_accs_l = []
    for run_name in runs_names:
        final_test_accs_l.extend(get_final_test_accs(run_name, runs, run_metrics_groups))

    mean_final_test_acc, std_final_test_acc = np.mean(final_test_accs_l), np.std(final_test_accs_l)

    print(f'final_test_accs_l: {final_test_accs_l}')

    print(f'mean_final_test_acc: {mean_final_test_acc}')
    print(f'std_final_test_acc: {std_final_test_acc}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='my-team/vfl-sandbox')
    args = parser.parse_args()

    n_clients = 4
    runs_names = [
                'local_cifar100_0.0_0.0_s0',
                'local_cifar100_0.0_0.0_s1',
                'local_cifar100_0.0_0.0_s2',
                'local_cifar100_0.0_0.0_s3',
                'local_cifar100_0.0_0.0_s4',
                ]

    main(runs_names, args.project_name, n_clients)
