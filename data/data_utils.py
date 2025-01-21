from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.custom_dataset import (CustomDataset, CustomMIMICDataset,
                                CreditDataset, HAPTDataset,
                                collate_fn, generate_mask_patterns_for_batches)


def get_dataloaders(args, config, p_miss_test=0.0):
    dataset, batch_size, num_workers = config["dataset"], config["batch_size"], config["num_workers"]
    data_dir = Path(__file__).absolute().parent.parent.parent / 'data' / dataset

    if dataset == 'hapt':
        train_ds = HAPTDataset(data_dir / 'X_train.txt', data_dir / 'y_train.txt', args.num_clients)
        test_ds  = HAPTDataset(data_dir / 'X_test.txt', data_dir / 'y_test.txt', args.num_clients)
        train_ld = create_data_loader(train_ds, batch_size, args.num_clients, args.p_miss_train, num_workers)
        test_ld  = create_data_loader(test_ds,  batch_size, args.num_clients, p_miss_test,    num_workers)
        return train_ld, test_ld

    elif dataset == 'credit':
        train_ds = CreditDataset(data_dir / 'X_train.npy', data_dir / 'y_train.npy', args.num_clients)
        labels = train_ds.y
        class_counts = np.array([len(np.where(labels == c)[0]) for c in np.unique(labels)])
        weight_per_class = 1.0 / class_counts
        sample_weights = np.array([weight_per_class[c] for c in labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_ld = create_data_loader(train_ds, batch_size, args.num_clients, args.p_miss_train, 
                                      num_workers=num_workers, sampler=sampler)
        
        test_ds = CreditDataset(data_dir / 'X_test.npy', data_dir / 'y_test.npy', args.num_clients)
        test_ld = create_data_loader(test_ds, batch_size, args.num_clients, p_miss_test, num_workers)
        return train_ld, test_ld

    elif dataset == 'mimic4':
        from models import mimic_model_utils
        vocab_d = mimic_model_utils.init(True, False, False, True, False, False)
        train_hids, test_hids = _create_mimic_train_test_split()
        mimic_train = CustomMIMICDataset(train_hids, vocab_d["gender_vocab"], vocab_d["eth_vocab"], 
                                         vocab_d["ins_vocab"], vocab_d["age_vocab"], "train")
        mimic_test  = CustomMIMICDataset(test_hids, vocab_d["gender_vocab"], vocab_d["eth_vocab"], 
                                         vocab_d["ins_vocab"], vocab_d["age_vocab"], "test")
        train_ld = create_data_loader(mimic_train, batch_size, args.num_clients, args.p_miss_train, 
                                      num_workers, drop_last=True)
        test_ld  = create_data_loader(mimic_test,  batch_size, args.num_clients, p_miss_test,    
                                      num_workers, drop_last=True)
        return train_ld, test_ld

    else:
        ds_map = {'mnist': datasets.MNIST, 'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100}
        if dataset not in ds_map:
            raise ValueError(f"Unknown dataset '{dataset}'.")
        ds_class = ds_map[dataset]
        transform = get_image_transforms(dataset)

        train_set = ds_class(data_dir, download=True, train=True, transform=transform)
        test_set  = ds_class(data_dir, download=True, train=False, transform=transform)
        train_ld  = create_data_loader(train_set, batch_size, args.num_clients, args.p_miss_train, num_workers)
        test_ld   = create_data_loader(test_set,  batch_size, args.num_clients, p_miss_test,    num_workers)
        return train_ld, test_ld


def create_data_loader(base_dataset, batch_size, num_clients, p_miss, num_workers=0, drop_last=False, sampler=None):
    num_samples = len(base_dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    p_observed = None if p_miss is None else (1 - p_miss)
    patterns = generate_mask_patterns_for_batches(num_batches, num_clients, p_observed)
    wrapped = CustomDataset(base_dataset, patterns, batch_size)
    return DataLoader(wrapped, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, 
                      drop_last=drop_last, sampler=sampler)


def get_image_transforms(dataset_name):
    if dataset_name == 'cifar100':
        norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        return transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), 
                                   norm])
    elif dataset_name == 'cifar10':
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        return transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), 
                                   norm])
    elif dataset_name == 'mnist':
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])
    else:
        raise ValueError(f"No transforms defined for '{dataset_name}'.")


def _create_mimic_train_test_split():
        labels = pd.read_csv('data/MIMIC-IV-Data-Pipeline/data/csv/labels.csv', header=0)

        hids = labels.iloc[:, 0]
        y = labels.iloc[:, 1]

        oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)
        hids = np.asarray(hids).reshape(-1, 1)
        hids, y = oversample.fit_resample(hids, y)
        hids = hids[:, 0]

        train_hids, test_hids = train_test_split(hids, test_size=0.2, random_state=42)

        return train_hids, test_hids
