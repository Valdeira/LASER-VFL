import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import torch
import pandas as pd


def _get_mask_per_batch(num_batches, num_blocks, p_miss=None):
    
    p_observed = None if p_miss is None else (1 - p_miss)
    if isinstance(p_observed, float):
        p_observed_array = np.array([p_observed] * num_blocks)
    elif p_observed is None:
        p_observed_array = np.random.beta(2.0, 2.0, num_blocks)

    patterns = [np.array([bool(int(x)) for x in bin(i)[2:].zfill(num_blocks)]) for i in range(2**num_blocks)]
    
    probabilities = []
    for pattern in patterns:
        prob = 1.0
        for block, p_observed in zip(pattern, p_observed_array):
            prob *= p_observed if block else (1 - p_observed)
        probabilities.append(prob)

    chosen_patterns = np.random.choice(len(patterns), size=num_batches, p=np.array(probabilities) / np.sum(probabilities))
    batch_patterns = [patterns[i] for i in chosen_patterns]

    return batch_patterns, p_observed_array


def collate_fn(batch):
    *features, labels, masks = zip(*batch)
    stacked_features = [torch.stack(feature_set) for feature_set in features]
    labels = torch.tensor(labels)
    mask = masks[0] # use the first mask (assuming same for all in the batch)
    return (*stacked_features, labels, mask)


class CustomDataset(Dataset):
    def __init__(self, base_dataset, batch_size, num_clients, p_miss):
        self.dataset = base_dataset
        num_samples = len(base_dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, num_clients, p_miss)
        self.batch_size = batch_size
        self.classes = base_dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        data = self.dataset[idx]

        features, label = data[:-1], data[-1]

        mask = self.batch_patterns[batch_idx]
        return (*features, label, torch.tensor(mask, dtype=torch.bool))


class CustomMIMICDataset(Dataset):
    def __init__(self, ids, gender_vocab, eth_vocab, ins_vocab, age_vocab, split_type):
        self.ids = ids
        self.gender_vocab = gender_vocab
        self.eth_vocab = eth_vocab
        self.ins_vocab = ins_vocab
        self.age_vocab = age_vocab
        self.classes = ['survived', 'died']

        self.labels = pd.read_csv('data/MIMIC-IV-Data-Pipeline/data/csv/labels.csv', header=0)
        self.cache_file = f'data/MIMIC-IV-Data-Pipeline/data/cached/cached_{split_type}_data.pkl'

        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            print("Loaded cached data from file.")
        else:
            self.cached_data = {}
            for sample in self.ids:
                dyn = pd.read_csv(f'data/MIMIC-IV-Data-Pipeline/data/csv/{sample}/dynamic.csv', header=[0, 1])
                dyn_temp = dyn['CHART'].to_numpy()
                dyn_tensor = torch.tensor(dyn_temp, dtype=torch.long)

                stat_tensor = torch.tensor(pd.read_csv(f'data/MIMIC-IV-Data-Pipeline/data/csv/{sample}/static.csv', header=[0, 1])['COND'].to_numpy(), dtype=torch.long)

                demo = pd.read_csv(f'data/MIMIC-IV-Data-Pipeline/data/csv/{sample}/demo.csv', header=0)[['gender', 'ethnicity', 'insurance', 'Age']]
                demo.replace({"gender": self.gender_vocab, "ethnicity": self.eth_vocab, "insurance": self.ins_vocab, "Age": self.age_vocab}, inplace=True)
                demo_tensor = torch.tensor(demo.values, dtype=torch.long)

                self.cached_data[sample] = {'dyn_tensor': dyn_tensor, 'stat_tensor': stat_tensor, 'demo_tensor': demo_tensor}

            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cached_data, f)
            print("Cached data saved to file.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.ids[index]

        cached_sample = self.cached_data[sample]
        dyn_tensor = cached_sample['dyn_tensor']
        stat_tensor = cached_sample['stat_tensor'].squeeze()
        demo_tensor = cached_sample['demo_tensor'].squeeze()

        label_row = self.labels.loc[self.labels['stay_id'] == sample, 'label']
        y = int(label_row.values[0]) if not label_row.empty else 0

        return dyn_tensor, stat_tensor, demo_tensor, torch.tensor(y)


class HAPTDataset(Dataset):
    def __init__(self, X_file, y_file, num_clients=None, feature_indices=None):
        # Load data from the provided files
        X = np.loadtxt(X_file).astype(float)
        y = np.loadtxt(y_file).astype(float)

        # Shuffle the samples along axis 0 (samples dimension)
        sample_indices = np.random.permutation(X.shape[0])
        X, y = X[sample_indices], y[sample_indices]

        # Adjust the number of features based on the specified number of clients
        if num_clients is not None:
            num_features = X.shape[1]
            new_num_features = num_features - (num_features % num_clients)
            X = X[:, :new_num_features]

        # Adjust labels to be zero-based by subtracting 1
        y = y - 1

        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        # Set up class names if needed
        self.classes = [f"{i}." for i in range(1, 13)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CreditDataset(Dataset):
    def __init__(self, X_file, y_file, num_clients=None):
        # Load data from the provided files
        X = np.load(X_file)
        y = np.load(y_file)

        # Adjust the number of features based on the specified number of clients
        if num_clients is not None:
            num_features = X.shape[1]
            new_num_features = num_features - (num_features % num_clients)
            X = X[:, :new_num_features]

        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        self.classes = [f"{i}." for i in [0, 1]]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
