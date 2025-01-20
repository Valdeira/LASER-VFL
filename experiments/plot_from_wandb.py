import wandb
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from utils import powerset_except_empty

def plot_trajectories(target_runs_names, traj_mean_d, traj_std_d, y_label, file_name, n_runs):
    
    plt.figure(figsize=(10, 6))
    labels = [f'{k}/4 blocks' for k in range(1, 4 + 1)]
    min_epoch, max_epoch = float('inf'), -float('inf')

    color_l = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ls_l = ['-', '--', '-.', ':']
    for target_run_name, color in zip(target_runs_names, color_l):
        traj_mean = traj_mean_d[target_run_name]
        traj_std = traj_std_d[target_run_name]
        for mean_df, std_df, label, ls in zip(traj_mean, traj_std, labels, ls_l):
            epochs = mean_df.index
            mean_acc = mean_df.iloc[:]
            std_acc = std_df.iloc[:]
            min_epoch = min(min_epoch, epochs.min())
            max_epoch = max(max_epoch, epochs.max())

            # remove seed and dataset from label
            first_underscore_index = target_run_name.find('_')
            run_name = target_run_name[first_underscore_index + 1:-6] # remove (plot) from the end
            dataset = target_run_name[:first_underscore_index]

            plt.plot(epochs, mean_acc, label=run_name + f' ({label})', ls=ls, color=color)
            plt.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color=color)

    plt.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')  # Major grid
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray')  # Minor grid
    
    if 'loss' in y_label:
        plt.yscale('log')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.title(y_label + f' for different subsets of clients ({dataset})', fontsize=16)
    plt.legend(loc='best', fontsize=11, ncol=n_runs)
    plt.xlim(min_epoch, max_epoch)
    if 'acc' in y_label:
        plt.ylim(0, 100)
    plt.savefig('experiments/results/' + file_name + '.pdf')
    plt.close()

def get_trajectories(run_name, runs, run_metrics_groups):
    
    for run in runs:
        if run.name == run_name:
            break
    else:
        print("No run found with that name.")

    run_group_dfs = []
    for run_metrics_group in run_metrics_groups:
        data = [row for row in run.scan_history(keys=run_metrics_group)]
        run_group_dfs.append(pd.DataFrame(data))

    run_group_test_acc_dfs = [df.filter(like='test_acc') for df in run_group_dfs]
    run_group_train_loss_dfs = [df.filter(like='train_loss') for df in run_group_dfs]

    run_group_test_acc_mean = []
    run_group_test_acc_std = []
    for run_group_test_acc_df in run_group_test_acc_dfs:
        # NOTE below we compute the mean and std test_acc for multiple subsets of blocks for the same seed
        # TODO allow for multiple seeds (compute mean and std across subsets AND seeds)
        run_group_test_acc_mean.append(run_group_test_acc_df.mean(axis=1))
        run_group_test_acc_std.append(run_group_test_acc_df.std(axis=1))

    run_group_train_loss_mean = []
    run_group_train_loss_std = []
    for run_group_train_loss_df in run_group_train_loss_dfs:
        run_group_train_loss_mean.append(run_group_train_loss_df.mean(axis=1))
        run_group_train_loss_std.append(run_group_train_loss_df.std(axis=1))
    
    return run_group_test_acc_mean, run_group_test_acc_std, run_group_train_loss_mean, run_group_train_loss_std

def main(runs_names, plot_name, wandb_project, n_clients, n_runs):

    run_metrics_groups = [[], [], [], []] # [sets with one client, ..., sets with n_clients clients]
    for run_prefix in powerset_except_empty(n_clients):
        base_name = str(run_prefix)
        run_metrics_groups[len(run_prefix) - 1].extend([base_name + '.train_loss', base_name + '.train_acc', base_name + '.test_loss', base_name + '.test_acc'])

    api = wandb.Api()
    runs = api.runs(path=wandb_project)
    
    mean_acc_d, std_acc_d = {}, {}
    mean_loss_d, std_loss_d = {}, {}
    for run_name in runs_names:
        mean_acc_d[run_name], std_acc_d[run_name], mean_loss_d[run_name], std_loss_d[run_name] = get_trajectories(run_name, runs, run_metrics_groups)
    
    plot_trajectories(runs_names, mean_acc_d, std_acc_d, 'Test accuracy', plot_name + '_test_acc', n_runs)
    plot_trajectories(runs_names, mean_loss_d, std_loss_d, 'Train loss', plot_name + '_train_loss', n_runs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default='my-team/vfl-sandbox')
    args = parser.parse_args()

    n_clients = 4
    plot_name = 'cifar10_missing_blocks'
    runs_names = ['cifar10_0.0_0.0_s0'] # , 'cifar10_0.0_0.0_s1',...

    main(runs_names, plot_name, args.project_name, n_clients, len(runs_names))
