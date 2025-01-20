import time
import random
import math
import wandb
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from decorator import decorator
from itertools import chain, combinations
from pathlib import Path
import yaml
import ast
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from data.custom_dataset import generate_mask_patterns_for_batches, CustomDataset, collate_fn
from data.custom_dataset import CustomMIMICDataset, HAPTDataset, CreditDataset


def get_dataloaders(args, config, p_miss_test=0.0):
    dataset, batch_size, num_workers = config["dataset"], config["batch_size"], config["num_workers"]

    data_dir = Path(__file__).absolute().parent.parent / 'data' / dataset

    if dataset in ['cifar10', 'cifar100']:
        if dataset == 'cifar100':
            transform_aux = transforms.Compose([
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                                ])
        if dataset == 'cifar10':
            transform_aux = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transform_aux,
                                        ])

    elif dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    elif dataset == 'hapt':

        train_dataset = HAPTDataset('../data/HAPT/X_train.txt', '../data/HAPT/y_train.txt', num_clients=args.num_clients)
        num_train_samples = len(train_dataset)
        num_train_batches = (num_train_samples + batch_size - 1) // batch_size
        p_observed_train = 1 - args.p_miss_train if args.p_miss_train is not None else None
        train_batch_patterns = generate_mask_patterns_for_batches(num_train_batches, args.num_clients, p_observed_train)
        train_dataset = CustomDataset(train_dataset, train_batch_patterns, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        test_dataset = HAPTDataset('../data/HAPT/X_test.txt', '../data/HAPT/y_test.txt', num_clients=args.num_clients)
        num_test_samples = len(test_dataset)
        num_test_batches = (num_test_samples + batch_size - 1) // batch_size
        p_observed_test = 1 - p_miss_test
        test_batch_patterns = generate_mask_patterns_for_batches(num_test_batches, args.num_clients, p_observed_test)
        test_dataset = CustomDataset(test_dataset, test_batch_patterns, batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        return train_loader, test_loader
    
    elif dataset == 'credit':

        train_dataset = CreditDataset('../data/credit/X_train.npy', '../data/credit/y_train.npy', num_clients=args.num_clients)
        num_train_samples = len(train_dataset)

        labels = train_dataset.y
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight_per_class = 1. / class_sample_count
        weights = np.array([weight_per_class[t] for t in labels])

        sampler = WeightedRandomSampler(weights, num_samples=num_train_samples, replacement=True)

        num_train_batches = (num_train_samples + batch_size - 1) // batch_size
        p_observed_train = 1 - args.p_miss_train if args.p_miss_train is not None else None
        train_batch_patterns = generate_mask_patterns_for_batches(num_train_batches, args.num_clients, p_observed_train)

        train_dataset = CustomDataset(train_dataset, train_batch_patterns, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)

        test_dataset = CreditDataset('../data/credit/X_test.npy', '../data/credit/y_test.npy', num_clients=args.num_clients)
        num_test_samples = len(test_dataset)
        num_test_batches = (num_test_samples + batch_size - 1) // batch_size
        p_observed_test = 1 - p_miss_test
        test_batch_patterns = generate_mask_patterns_for_batches(num_test_batches, args.num_clients, p_observed_test)
        test_dataset = CustomDataset(test_dataset, test_batch_patterns, batch_size)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        return train_loader, test_loader

    elif dataset == 'mimic4':
        from models import mimic_model_utils
        vocab_d = mimic_model_utils.init(True, False, False, True, False, False)
        train_hids, test_hids = create_mimic_train_test_split()
        
        train_dataset = CustomMIMICDataset(train_hids, vocab_d["gender_vocab"], vocab_d["eth_vocab"], vocab_d["ins_vocab"], vocab_d["age_vocab"], "train")
        num_train_samples = len(train_dataset)
        num_train_batches = (num_train_samples + batch_size - 1) // batch_size
        p_observed_train = 1 - args.p_miss_train if args.p_miss_train is not None else None
        train_batch_patterns = generate_mask_patterns_for_batches(num_train_batches, args.num_clients, p_observed_train)
        train_dataset = CustomDataset(train_dataset, train_batch_patterns, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
        
        test_dataset = CustomMIMICDataset(test_hids, vocab_d["gender_vocab"], vocab_d["eth_vocab"], vocab_d["ins_vocab"], vocab_d["age_vocab"], "test")
        num_test_samples = len(test_dataset)
        num_test_batches = (num_test_samples + batch_size - 1) // batch_size
        p_observed_test = 1 - p_miss_test
        test_batch_patterns = generate_mask_patterns_for_batches(num_test_batches, args.num_clients, p_observed_test)
        test_dataset = CustomDataset(test_dataset, test_batch_patterns, batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
        
        return train_loader, test_loader
    
    datasets_dict = {'mnist': datasets.MNIST, 'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}
    Dataset = datasets_dict[dataset]
    
    train_set = Dataset(data_dir, download=True, train=True, transform=transform)
    num_train_samples = len(train_set)
    num_train_batches = (num_train_samples + batch_size - 1) // batch_size
    p_observed_train = 1 - args.p_miss_train if args.p_miss_train is not None else None
    train_batch_patterns = generate_mask_patterns_for_batches(num_train_batches, args.num_clients, p_observed_train)
    train_dataset = CustomDataset(train_set, train_batch_patterns, batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    test_set = Dataset(data_dir, download=True, train=False, transform=transform)
    num_test_samples = len(test_set)
    num_test_batches = (num_test_samples + batch_size - 1) // batch_size
    p_observed_test = 1 - p_miss_test
    test_batch_patterns = generate_mask_patterns_for_batches(num_test_batches, args.num_clients, p_observed_test)
    test_dataset = CustomDataset(test_set, test_batch_patterns, batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, test_loader

def print_exp_info(args, config, epoch):
    s = f'epoch:[{epoch + 1}/{config["num_epochs"]}]  {args.device}  {args.task_name}  method:{args.method}  K:{args.num_clients}  p_miss_train:{args.p_miss_train}  seed:{args.seed}  lr:{config["lr"]}  decay:{config["weight_decay"]}  mom:{config["momentum"]}'
    print(f"{len(s) * '-'}\n{s}\n{len(s) * '-'}")

def init_wandb(args, config):
    
    wandb_config = {
            'dataset': config["dataset"],
            'architecture': config["model"],
            'cuda': args.cuda_id,
            'method': args.method,
            'lr': config["lr"],
            'n_epochs': config["num_epochs"],
            'seed': args.seed,
            'num_clients': args.num_clients,
            'batch_size': config["batch_size"],
            'weight_decay': config["weight_decay"],
            'momentum': config["momentum"],
            }

    name = args.wandb_name if args.wandb_name is not None else f'{args.task_name}_{args.method}_K{args.num_clients}_p_miss_train{args.p_miss_train}_s{args.seed}'

    wandb.init(project=args.project, config=wandb_config, name=name)

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

@decorator
def time_decorator(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start
    print(f"time to run {func.__name__}(): {elapsed_time:.2f}s.")
    return result

def powerset_except_empty(n):
    s = list(range(n))
    return list(chain.from_iterable(combinations(s, r) for r in range(1, n+1)))

@time_decorator
def test_decoupled(dataloader, models, criterion, device, is_final=False, compute_f1=False, is_train_data=False):
    
    for model in models:
        model.eval()
    
    num_models = len(models)
    is_powerset = True if num_models == 2 ** model.num_clients - 1 else False
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l, correct_l = [0.0] * num_models, [0.0] * num_models
    if compute_f1:
        true_positive_l, false_positive_l, false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models
    if compute_f1 and is_final:
        final_true_positive_l, final_false_positive_l, final_false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models

    final_correct = 0.0
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)

            if torch.sum(mask).item() == 0:
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue

            for i, model in enumerate(models):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_l[i] += loss.item()
                predicted = outputs.argmax(1)
                correct_l[i] += (predicted == targets).float().sum().item()
                if compute_f1:
                    true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                
                if is_final:

                    # powerset of models
                    if is_powerset:
                        if set(model.clients_in_model) == set(torch.nonzero(mask).view(-1).tolist()):
                            final_correct += (predicted == targets).float().sum().item()
                            if compute_f1:
                                final_true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                                final_false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                                final_false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()

                    # local model
                    elif len(model.clients_in_model) == 1:
                        [client_in_model] = model.clients_in_model
                        observed_blocks_l = torch.nonzero(mask).view(-1).tolist()
                        if client_in_model in observed_blocks_l:
                            predicted = outputs.argmax(1)
                            final_correct += (predicted == targets).float().sum().item() / len(observed_blocks_l)
                            # assuming that when training local model we train one per each client
                            if compute_f1:
                                final_true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                                final_false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                                final_false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                    
                    # standard vfl model
                    elif len(model.clients_in_model) == len(mask):
                        if torch.all(mask):
                            predicted = outputs.argmax(1)
                        else:
                            predicted = torch.randint(0, outputs.shape[1], targets.shape).to(device) # random predictions
                        final_correct += (predicted == targets).float().sum().item()
                        if compute_f1:
                            final_true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                            final_false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                            final_false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                    
                    else:
                        raise NotImplementedError("Currently asssuming one of (1) local model, (2) standard VFL, or (3) powerset.")
                    

    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_split_type}_acc"] = 100 * final_correct / num_samples
        if compute_f1:
            final_f1_per_predictor = [get_f1(tp, fp, fn) for tp, fp, fn in zip(final_true_positive_l, final_false_positive_l, final_false_negative_l)]
            final_f1_per_predictor = [f1 for f1 in final_f1_per_predictor if f1 != 0.0] # remove models that were not used for prediction (do this more cleanly later using a dictionary)
            metrics[f"final_{data_split_type}_f1"] = sum(final_f1_per_predictor) / len(final_f1_per_predictor)
    else:
        avg_loss = [loss / num_batches for loss in loss_l]
        accuracy = [100 * correct / num_samples for correct in correct_l]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

@time_decorator
def train_decoupled(dataloader, models, optimizers, criterion, args, compute_f1=False):
    
    for model in models:
        model.train()
    
    num_models = len(models)
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l, correct_l = [0.0] * num_models, [0.0] * num_models
    if compute_f1:
        true_positive_l, false_positive_l, false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models
    num_samples_per_block_l = [num_samples] * num_models
    num_batches_per_block_l = [num_batches] * num_models

    for batch_num, batch in enumerate(dataloader):
        *inputs, targets, mask = batch
        inputs, targets = [tensor.to(args.device) for tensor in inputs], targets.to(args.device)

        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            
            # if not all the blocks needed for this model are observed, skip batch
            if not mask[torch.tensor(model.clients_in_model)].all().item():
                num_batches_per_block_l[i] -= 1
                num_samples_per_block_l[i] -= len(inputs)
                continue
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_l[i] += loss.item()
            predicted = outputs.argmax(1)
            correct_l[i] += (predicted == targets).float().sum().item()
            if compute_f1:
                true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
        
        if (batch_num + 1) % 25 == 0:
            try:
                loss_value = loss.item()
                print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): {loss_value:.4f}")
            except UnboundLocalError:
                print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): N/A")
            
            
    avg_loss = [loss / num_batches_per_block_l[i] for i, loss in enumerate(loss_l)]
    accuracy = [100 * correct / num_samples_per_block_l[i] for i, correct in enumerate(correct_l)]
    metrics = {"train_loss": avg_loss, "train_acc": accuracy}
    if compute_f1:
        metrics["train_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

@time_decorator
def test_laser(dataloader, models, criterion, device, is_final=False, compute_f1=False, is_train_data=False):
    
    [model] = models

    model.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)

    loss_d = {clients_l: 0.0 for clients_l in model.powerset}
    correct_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1:
        true_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_negative_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1 and is_final:
        final_true_positive_l = [0.0 for _ in range(model.num_clients)]
        final_false_positive_l = [0.0 for _ in range(model.num_clients)]
        final_false_negative_l = [0.0 for _ in range(model.num_clients)]

    final_correct = 0.0 
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)
            
            if torch.sum(mask).item() == 0:
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue

            outputs_per_head_l = model(inputs, training=False)
            for i, outputs_per_task_d in enumerate(outputs_per_head_l):
                for clients_subset, outputs in outputs_per_task_d.items():
                    loss = criterion(outputs, targets)
                    
                    # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
                    # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
                    norm_constant = len(clients_subset)
                    loss_d[clients_subset] += loss.item() / norm_constant
                    predicted = outputs.argmax(1)
                    correct_d[clients_subset] += (predicted == targets).float().sum().item() / norm_constant
                    if compute_f1:
                        # this is not the f1 of any given predictor, but rather the f1 obtain from the average
                        # TP, FP, and FN across the different predictors for the same set of blocks (diff from final result)
                        true_positive_d[clients_subset] += ((predicted == 1) & (targets == 1)).sum().item() / norm_constant
                        false_positive_d[clients_subset] += ((predicted == 1) & (targets == 0)).sum().item() / norm_constant
                        false_negative_d[clients_subset] += ((predicted == 0) & (targets == 1)).sum().item() / norm_constant

                    # check if final (after training) and if this task uses the (exact) set of observed blocks
                    if is_final and set(clients_subset) == set(torch.nonzero(mask).view(-1).tolist()):
                        final_correct += (predicted == targets).float().sum().item() / len(clients_subset)
                        if compute_f1:
                            final_true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                            final_false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                            final_false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_split_type}_acc"] = 100 * final_correct / num_samples
        if compute_f1:
            final_f1_per_predictor = [get_f1(tp, fp, fn) for tp, fp, fn in zip(final_true_positive_l, final_false_positive_l, final_false_negative_l)]
            metrics[f"final_{data_split_type}_f1"] = sum(final_f1_per_predictor) / len(final_f1_per_predictor)
    else:
        avg_loss = [loss / num_batches for loss in [loss_d[clients_l] for clients_l in model.powerset]]
        accuracy = [100 * correct / num_samples for correct in [correct_d[clients_l] for clients_l in model.powerset]]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            true_positive = [true_positive_d[clients_l] for clients_l in model.powerset]
            false_positive = [false_positive_d[clients_l] for clients_l in model.powerset]
            false_negative = [false_negative_d[clients_l] for clients_l in model.powerset]
            avg_loss = [loss / num_batches for loss in [loss_d[clients_l] for clients_l in model.powerset]]
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive, false_positive, false_negative)]

    return metrics

@time_decorator
def train_laser(dataloader, models, optimizers, criterion, args, compute_f1=False):
    
    [model], [optimizer] = models, optimizers

    device = args.device

    model.train()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    loss_d = {clients_l: 0.0 for clients_l in model.powerset}
    correct_d = {clients_l: 0.0 for clients_l in model.powerset}
    if compute_f1:
        true_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_positive_d = {clients_l: 0.0 for clients_l in model.powerset}
        false_negative_d = {clients_l: 0.0 for clients_l in model.powerset}
    
    for batch_num, batch in enumerate(dataloader):
        *inputs, targets, mask = batch
        inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)
        
        if torch.sum(mask).item() == 0:
            num_batches -= 1 # in practice, this batch is not used
            num_samples -= len(inputs[0]) # in practice, these samples are not used
            continue

        optimizer.zero_grad()

        observed_blocks = torch.nonzero(mask).view(-1).tolist()

        outputs_per_head_l = model(inputs, training=True, observed_blocks=observed_blocks)
        
        total_loss = 0
        for outputs_per_task_d in outputs_per_head_l:

            head_loss = 0
            for clients_subset, outputs in outputs_per_task_d.items():
                loss = criterion(outputs, targets)

                norm_constant = 1.0
                norm_constant = norm_constant / len(clients_subset) # account for multiple heads
                
                # if a given element (client) must be in a set, then we do n-1 choose k-1 instead
                n, k = (len(observed_blocks)-1, len(clients_subset)-1)
                norm_constant = norm_constant * math.comb(n, k)

                head_loss += loss * norm_constant

                # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
                # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
                loss_d[clients_subset] += loss.item() * norm_constant
                predicted = outputs.argmax(1)
                correct_d[clients_subset] += (predicted == targets).float().sum().item() * norm_constant
                if compute_f1:
                        # this is not the f1 of any given predictor, but rather the f1 obtain from the average
                        # TP, FP, and FN across the different predictors for the same set of blocks (diff from final result)
                        true_positive_d[clients_subset] += ((predicted == 1) & (targets == 1)).sum().item() / norm_constant
                        false_positive_d[clients_subset] += ((predicted == 1) & (targets == 0)).sum().item() / norm_constant
                        false_negative_d[clients_subset] += ((predicted == 0) & (targets == 1)).sum().item() / norm_constant

            total_loss += head_loss

        total_loss.backward()
        optimizer.step()
        
        if (batch_num + 1) % 25 == 0:
            print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): {loss.item():.4f}")
            
    loss_l = [loss_d[clients_l] for clients_l in model.powerset]
    correct_l = [correct_d[clients_l] for clients_l in model.powerset]

    avg_loss = [loss / num_batches for loss in loss_l]
    accuracy = [100 * correct / num_samples for correct in correct_l]
    metrics = {"train_loss": avg_loss, "train_acc": accuracy}
    if compute_f1:
        true_positive_l = [true_positive_d[clients_l] for clients_l in model.powerset]
        false_positive_l = [false_positive_d[clients_l] for clients_l in model.powerset]
        false_negative_l = [false_negative_d[clients_l] for clients_l in model.powerset]
        metrics["train_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

def random_argmax_along_axis(tensor, axis=1):
    
    if axis != 1:
        raise NotImplementedError("This function only supports axis=1")

    random_argmax_indices = []
    for row in tensor:
        max_value = torch.max(row)
        max_indices = torch.nonzero(row == max_value, as_tuple=False).squeeze()
        if len(max_indices.shape) > 0 and max_indices.shape[0] > 1:
            random_index = max_indices[random.choice(range(max_indices.shape[0]))]
        else:
            random_index = max_indices.item()
        random_argmax_indices.append(random_index)

    return torch.tensor(random_argmax_indices)

@time_decorator
def test_ensemble(dataloader, models, criterion, device, is_final=False, compute_f1=False, is_train_data=False):
    
    for model in models:
        model.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l = [0.0 for _ in range(len(models))]
    correct_l = [0.0 for _ in range(len(models))]
    if compute_f1:
        true_positive_l = [0.0 for _ in range(len(models))]
        false_positive_l = [0.0 for _ in range(len(models))]
        false_negative_l = [0.0 for _ in range(len(models))]
    
    final_correct = 0.0
    if compute_f1 and is_final:
        final_true_positive = 0.0
        final_false_positive = 0.0
        final_false_negative = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)

            if torch.sum(mask).item() == 0:
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue
            
            cur_batch_size = inputs[0].size(0)
            ensemble_votes = torch.zeros(cur_batch_size, len(dataloader.dataset.classes)).to(device)
            for i, model in enumerate(models):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_l[i] += loss.item()
                predicted = outputs.argmax(1)
                correct_l[i] += (predicted == targets).float().sum().item()
                if compute_f1:
                    true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                
                if is_final:
                    if len(model.clients_in_model) == 1: # local model
                        client_in_model = next(iter(model.clients_in_model))
                        observed_blocks_l = torch.nonzero(mask).view(-1).tolist()
                        if client_in_model in observed_blocks_l:
                            argmax_indices = outputs.argmax(1)
                            one_hot = torch.zeros_like(outputs).scatter_(1, argmax_indices.unsqueeze(1), 1)
                            ensemble_votes += one_hot
                    else:
                        raise NotImplementedError("The final accuracy with p_miss@test>=0 is only implemented for local models and standard VFL models.")
            
            if is_final:
                predicted = random_argmax_along_axis(ensemble_votes).to(device)
                final_correct += (predicted == targets).float().sum().item()
                # assuming that when training local model we train one per each client
                if compute_f1:
                    final_true_positive += ((predicted == 1) & (targets == 1)).sum().item()
                    final_false_positive += ((predicted == 1) & (targets == 0)).sum().item()
                    final_false_negative += ((predicted == 0) & (targets == 1)).sum().item()

    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_split_type}_acc"] = 100 * final_correct / num_samples
        if compute_f1:
            metrics[f"final_{data_split_type}_f1"] = get_f1(final_true_positive, final_false_positive, final_false_negative)
    else:
        avg_loss = [loss / num_batches for loss in loss_l]
        accuracy = [100 * correct / num_samples for correct in correct_l]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

@time_decorator
def test_plug(dataloader, models, criterion, device, is_final=False, compute_f1=False, is_train_data=False):
    
    for model in models:
        model.eval()
    
    num_models = len(models)
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l, correct_l = [0.0] * num_models, [0.0] * num_models
    if compute_f1:
        true_positive_l, false_positive_l, false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models
    if compute_f1 and is_final:
        final_true_positive_l, final_false_positive_l, final_false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models

    final_correct = 0.0
    with torch.no_grad():
        for batch in dataloader:
            *inputs, targets, mask = batch
            inputs, targets = [tensor.to(device) for tensor in inputs], targets.to(device)

            # if client zero (active party) unobserved, skip batch
            if not mask[-1].item():
                num_batches -= 1 # in practice, this batch is not used
                num_samples -= len(inputs[0]) # in practice, these samples are not used
                continue

            for i, model in enumerate(models):
                outputs = model(inputs, mask)
                loss = criterion(outputs, targets)
                loss_l[i] += loss.item()
                predicted = outputs.argmax(1)
                correct_l[i] += (predicted == targets).float().sum().item()
                if compute_f1:
                    true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                    false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                    false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                if is_final:
                    final_correct += (predicted == targets).float().sum().item()
                    if compute_f1:
                        final_true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                        final_false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                        final_false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
                    
    data_split_type = "train" if is_train_data else "test"
    metrics = {}
    if is_final:
        metrics[f"final_{data_split_type}_acc"] = 100 * final_correct / num_samples
        if compute_f1:
            final_f1_per_predictor = [get_f1(tp, fp, fn) for tp, fp, fn in zip(final_true_positive_l, final_false_positive_l, final_false_negative_l)]
            final_f1_per_predictor = [f1 for f1 in final_f1_per_predictor if f1 != 0.0] # remove models that were not used for prediction (do this more cleanly later using a dictionary)
            metrics[f"final_{data_split_type}_f1"] = sum(final_f1_per_predictor) / len(final_f1_per_predictor)
    else:
        avg_loss = [loss / num_batches for loss in loss_l]
        accuracy = [100 * correct / num_samples for correct in correct_l]
        metrics[f"{data_split_type}_loss"] = avg_loss
        metrics[f"{data_split_type}_acc"] = accuracy
        if compute_f1:
            metrics[f"{data_split_type}_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

@time_decorator
def train_plug(dataloader, models, optimizers, criterion, args, compute_f1=False):
    
    for model in models:
        model.train()
    
    num_models = len(models)
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_l, correct_l = [0.0] * num_models, [0.0] * num_models
    if compute_f1:
        true_positive_l, false_positive_l, false_negative_l = [0.0] * num_models, [0.0] * num_models, [0.0] * num_models
    num_samples_per_block_l = [num_samples] * num_models
    num_batches_per_block_l = [num_batches] * num_models

    for batch_num, batch in enumerate(dataloader):
        *inputs, targets, mask = batch
        inputs, targets = [tensor.to(args.device) for tensor in inputs], targets.to(args.device)

        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            
            # if client zero (active party) unobserved, skip batch
            if not mask[-1].item():
                num_batches_per_block_l[i] -= 1
                num_samples_per_block_l[i] -= len(inputs)
                continue

            optimizer.zero_grad()
            outputs = model(inputs, mask, args.p_drop)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_l[i] += loss.item()
            predicted = outputs.argmax(1)
            correct_l[i] += (predicted == targets).float().sum().item()
            if compute_f1:
                true_positive_l[i] += ((predicted == 1) & (targets == 1)).sum().item()
                false_positive_l[i] += ((predicted == 1) & (targets == 0)).sum().item()
                false_negative_l[i] += ((predicted == 0) & (targets == 1)).sum().item()
        
        if (batch_num + 1) % 25 == 0:
            try:
                loss_value = loss.item()
                print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): {loss_value:.4f}")
            except UnboundLocalError:
                print(f"\tBatch [{batch_num + 1}/{len(dataloader)}] train loss (last): N/A")
            
            
    avg_loss = [loss / num_batches_per_block_l[i] for i, loss in enumerate(loss_l)]
    accuracy = [100 * correct / num_samples_per_block_l[i] for i, correct in enumerate(correct_l)]
    metrics = {"train_loss": avg_loss, "train_acc": accuracy}
    if compute_f1:
        metrics["train_f1"] = [get_f1(tp, fp, fn) for tp, fp, fn in zip(true_positive_l, false_positive_l, false_negative_l)]

    return metrics

def setup_task(args):

    from models import get_model
    from optimizers import get_optimizer
    from criterions import get_criterion
    from schedulers import get_scheduler
    
    with open("configs/task_config.yaml", "r") as file:
        configurations = yaml.safe_load(file)

    config = configurations[args.method][args.task_name][args.num_clients][args.p_miss_train]

    model = get_model(args.method, config["model"], config["dataset"], args, config)
    optimizer = get_optimizer(args.method, config["optimizer"], model, config)
    scheduler = get_scheduler(args.method, config["scheduler"], optimizer, config)
    criterion = get_criterion(config["criterion"])

    if args.method == "laser":
        train, test = train_laser, test_laser
    elif args.method == "decoupled":
        train, test = train_decoupled, test_decoupled
    elif args.method == "ensemble":
        train, test = train_decoupled, test_ensemble
    elif args.method == "plug":
        train, test = train_plug, test_plug
    
    return config, model, optimizer, scheduler, criterion, train, test

def check_sets_of_clients_valid(args):

        if args.blocks_in_tasks_t is not None:
            if args.method == 'laser':
                raise ValueError("blocks_in_tasks_t should not be provided when method is 'laser'")
            try: # Convert the string representation of the list of tuples to an actual list of tuples
                args.blocks_in_tasks_t = ast.literal_eval(args.blocks_in_tasks_t)
            except (ValueError, SyntaxError):
                raise ValueError("Invalid format for blocks_in_tasks_t. It should be a valid list of tuples.")
        else:
            # Default to powerset_except_empty(args.num_clients) for both 'decoupled' and 'laser'
            args.blocks_in_tasks_t = powerset_except_empty(args.num_clients)

def create_mimic_train_test_split():
        labels = pd.read_csv('data/MIMIC-IV-Data-Pipeline/data/csv/labels.csv', header=0)

        hids = labels.iloc[:, 0]
        y = labels.iloc[:, 1]
        # print(f"(Original) Total Samples: {len(hids)} | Positive Samples: {y.sum()}")

        oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
        hids = np.asarray(hids).reshape(-1, 1)
        hids, y = oversample.fit_resample(hids, y)
        hids = hids[:, 0]
        # print(f"(Oversampled) Total Samples: {len(hids)} | Positive Samples: {y.sum()}")

        train_hids, test_hids = train_test_split(hids, test_size=0.2, random_state=42)

        return train_hids, test_hids

def get_f1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1

def get_metrics(train_metrics, test_metrics, compute_f1, blocks_in_tasks_t):
    
    metrics = {
            str(clients): {
                "train_loss": train_metrics["train_loss"][i],
                "train_acc": train_metrics["train_acc"][i],
                "test_loss": test_metrics["test_loss"][i],
                "test_acc": test_metrics["test_acc"][i],
                **({"train_f1": train_metrics["train_f1"][i], "test_f1": test_metrics["test_f1"][i]} if compute_f1 else {})
            }
            for i, clients in enumerate(blocks_in_tasks_t)
        }
    
    return metrics
