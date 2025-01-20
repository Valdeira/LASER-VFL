import random
import torch
from torch import nn
from torch.autograd import *

from utils import powerset_except_empty


class FeatureExtractor(nn.Module):
    def __init__(self, vocab_d, batch_size, cut_dim, n_chart_features, n_cond_features, demo_features_idx):
        super().__init__()
        self.batch_size = batch_size
        self.vocab_sizes = {
            'cond': vocab_d["cond_vocab_size"], # n_cond_features = self.vocab_sizes['cond'] # 1419
            'chart': vocab_d["chart_vocab_size"], # n_chart_features = self.vocab_sizes['chart'] # 449
            'ethnicity': len(vocab_d["eth_vocab"]),
            'gender': len(vocab_d["gender_vocab"]),
            'age': len(vocab_d["age_vocab"]),
            'insurance': len(vocab_d["ins_vocab"])
        }

        global_demo_features_l = ['gender', 'ethnicity', 'insurance', 'age']
        self.demo_features_idx = demo_features_idx
        self.demo_features = [global_demo_features_l[i] for i in demo_features_idx]
        n_demo_features = len(self.demo_features)

        self.modalities = 2
        self.num_layers = 1
        self.latent_size = 128
        self.embed_size = 32
        self.hidden_size = 128
        
        self.chart_embed = ValEmbed(n_chart_features, self.embed_size, self.latent_size)
        self.cond_embed = StatEmbed(n_cond_features, self.embed_size, self.latent_size)
        
        if 'ethnicity' in self.demo_features:
            self.ethEmbed = nn.Embedding(self.vocab_sizes['ethnicity'], self.latent_size)
        if 'gender' in self.demo_features:
            self.genderEmbed = nn.Embedding(self.vocab_sizes['gender'], self.latent_size)
        if 'age' in self.demo_features:
            self.ageEmbed = nn.Embedding(self.vocab_sizes['age'], self.latent_size)
        if 'insurance' in self.demo_features:
            self.insEmbed = nn.Embedding(self.vocab_sizes['insurance'], self.latent_size)
        
        self.embedfc = nn.Linear(self.latent_size * (self.modalities + n_demo_features), self.latent_size)
        self.lstm = nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, cut_dim)

    def forward(self, inputs):
        chart, conds, demo = inputs
        batch_size = min(self.batch_size, chart.shape[0], conds.shape[0])
        output = torch.zeros(size=(0, 0))

        embedded_chart = self.chart_embed(chart[-batch_size:])
        output = embedded_chart if not output.nelement() else torch.cat((output, embedded_chart), 2)

        embedded_condition = self.cond_embed(conds[-batch_size:]).unsqueeze(1).repeat(1, output.shape[1], 1) # conds[-batch_size:] was conds[-batch_size:].to(self.device)
        output = torch.cat((output, embedded_condition), 2)

        potential_embeddings = {
                                'gender': getattr(self, 'genderEmbed', None),
                                'ethnicity': getattr(self, 'ethEmbed', None),
                                'insurance': getattr(self, 'insEmbed', None),
                                'age': getattr(self, 'ageEmbed', None)
                                }
        demographic_embeddings = {key: value for key, value in potential_embeddings.items() if key in self.demo_features and value is not None}
        for local_idx, demo_name in enumerate(self.demo_features):
            data = demo[:, local_idx][:batch_size].long()
            embedded_demographic = demographic_embeddings[demo_name](data).unsqueeze(1).repeat(1, output.shape[1], 1)
            output = torch.cat((output, embedded_demographic), 2)

        output = self.embedfc(output)
        _, (output, _) = self.lstm(output)
        output = output[-1].squeeze()
        output = self.fc(output)
        
        return output

class DecoupledModel(nn.Module):

    def __init__(self, dataset, args, vocab_d, config, clients_in_model=None, aggregation="mean"):
        
        super().__init__()

        if dataset == 'mimic4':
            num_classes = 2
        else:
            raise ValueError(f"Unexpected dataset {dataset}")
        
        assert args.num_clients == 4

        self.num_clients = args.num_clients
        self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        cut_dim = 64
        # TODO get from somewhere (don't hardcode)
        n_chart_features = 449
        n_cond_features = 1419
        chart_features_idx_l = list(range(n_chart_features))
        chart_features_idx_l = split_into_chunks(chart_features_idx_l, args.num_clients)
        self.chart_features_idx_l = chart_features_idx_l
        cond_features_idx_l = list(range(n_cond_features))
        cond_features_idx_l = split_into_chunks(cond_features_idx_l, args.num_clients)
        self.cond_features_idx_l = cond_features_idx_l
        demo_features_l = ['gender', 'ethnicity', 'insurance', 'age']
        demo_features_l = split_into_chunks(demo_features_l, args.num_clients)
        demo_features_idx_l = list(range(len(demo_features_l)))
        demo_features_idx_l = split_into_chunks(demo_features_idx_l, args.num_clients)
        self.demo_features_idx_l = demo_features_idx_l
        self.demo_features_l = demo_features_l
        
        self.feature_extractors = nn.ModuleList()
        for block_idx, (chart_features_idx, cond_features_idx, demo_features_idx) in enumerate(zip(chart_features_idx_l, cond_features_idx_l, demo_features_idx_l)):
            if block_idx in self.clients_in_model:    
                n_chart_features = len(chart_features_idx)
                n_cond_features = len(cond_features_idx)
                self.feature_extractors.append(FeatureExtractor(vocab_d, config["batch_size"], cut_dim, n_chart_features, n_cond_features, demo_features_idx))
        
        self.fusion_model = FusionModel(cut_dim, num_classes, aggregation, args.num_clients)

    def get_block(self, x, index):
        
        chart, conds, demo = x
        
        chart_indices = torch.tensor(self.chart_features_idx_l[index], dtype=torch.long).to(chart.device)
        cond_indices = torch.tensor(self.cond_features_idx_l[index], dtype=torch.long).to(chart.device)
        demo_indices = torch.tensor(self.demo_features_idx_l[index], dtype=torch.long).to(chart.device)

        chart_subset = chart.index_select(dim=-1, index=chart_indices)
        conds_subset = conds.index_select(dim=-1, index=cond_indices)
        demo_subset = demo.index_select(dim=-1, index=demo_indices)
        
        return chart_subset, conds_subset, demo_subset
    
    def forward(self, x, plug_mask=None, p_drop=0):
        
        if plug_mask is not None: # for PlugVFL we use mask in the forward pass
            """
            missing feature blocks are always zeros at the fusion model;
            during training, p_drop>0 is provided and the observed feature blocks
            can also  be dropped, leading to zeros at the fusion model w.p. p_drop in (0,0.5)
            """
            new_mask = drop_mask(plug_mask, p_drop)
            embeddings = [
                self.feature_extractors[i](self.get_block(x, j)) if new_mask[i] else 
                torch.zeros_like(self._get_dummy_output(self.feature_extractors[i], self.get_block(x, j)))
                for i, j in enumerate(self.clients_in_model)
            ]
            return self.fusion_model(embeddings)
        else:
            embeddings = [self.feature_extractors[i](self.get_block(x, j)) for i, j in enumerate(self.clients_in_model)]
            return self.fusion_model(embeddings)
    
    def _get_dummy_output(self, feature_extractor, input_tensor):
        with torch.no_grad():
            return feature_extractor(input_tensor)

class LaserModel(nn.Module):
    
    def __init__(self, dataset, args, vocab_d, config):
        
        super().__init__()

        if dataset == 'mimic4':
            num_classes = 2
        else:
            raise ValueError(f"Unexpected dataset {dataset}")
        
        assert args.num_clients == 4

        self.powerset = powerset_except_empty(args.num_clients)
        self.args = args
        self.num_clients = args.num_clients

        # self.clients_in_model = clients_in_model if clients_in_model is not None else list(range(args.num_clients))

        cut_dim = 64
        # TODO get from somewhere (don't hardcode)
        n_chart_features = 449
        n_cond_features = 1419
        chart_features_idx_l = list(range(n_chart_features))
        chart_features_idx_l = split_into_chunks(chart_features_idx_l, args.num_clients)
        self.chart_features_idx_l = chart_features_idx_l
        cond_features_idx_l = list(range(n_cond_features))
        cond_features_idx_l = split_into_chunks(cond_features_idx_l, args.num_clients)
        self.cond_features_idx_l = cond_features_idx_l
        demo_features_l = ['gender', 'ethnicity', 'insurance', 'age']
        demo_features_l = split_into_chunks(demo_features_l, args.num_clients)
        demo_features_idx_l = list(range(len(demo_features_l)))
        demo_features_idx_l = split_into_chunks(demo_features_idx_l, args.num_clients)
        self.demo_features_idx_l = demo_features_idx_l
        self.demo_features_l = demo_features_l
        
        self.feature_extractors = nn.ModuleList()
        for block_idx, (chart_features_idx, cond_features_idx, demo_features_idx) in enumerate(zip(chart_features_idx_l, cond_features_idx_l, demo_features_idx_l)):
            if block_idx in range(args.num_clients):    
                n_chart_features = len(chart_features_idx)
                n_cond_features = len(cond_features_idx)
                self.feature_extractors.append(FeatureExtractor(vocab_d, config["batch_size"], cut_dim, n_chart_features, n_cond_features, demo_features_idx))
        
        self.fusion_models = nn.ModuleList([FusionModel(cut_dim, num_classes) for _ in range(args.num_clients)])
    
    def get_block(self, x, index):
        
        chart, conds, demo = x
        
        chart_indices = torch.tensor(self.chart_features_idx_l[index], dtype=torch.long).to(chart.device)
        cond_indices = torch.tensor(self.cond_features_idx_l[index], dtype=torch.long).to(chart.device)
        demo_indices = torch.tensor(self.demo_features_idx_l[index], dtype=torch.long).to(chart.device)

        chart_subset = chart.index_select(dim=-1, index=chart_indices)
        conds_subset = conds.index_select(dim=-1, index=cond_indices)
        demo_subset = demo.index_select(dim=-1, index=demo_indices)
        
        return chart_subset, conds_subset, demo_subset
    
    def forward(self, x, training=True, observed_blocks=None):

        if observed_blocks is None: # NOTE this is be the case e.g. for test data
            observed_blocks = list(range(self.num_clients))
        
        embeddings = {}
        for i in observed_blocks:
            local_input = self.get_block(x, i)
            embeddings[i] = self.feature_extractors[i](local_input)

        if training:
            outputs = []
            for i, fusion_model in enumerate(self.fusion_models):
                if i not in observed_blocks:
                    continue
                sets_considered_by_head = [clients_l for clients_l in self.powerset if i in clients_l and set(clients_l).issubset(set(observed_blocks))]
                head_output = {}
                for num_clients_in_agg in range(1, len(observed_blocks) + 1):
                    set_to_sample = [client_set for client_set in sets_considered_by_head if len(client_set) == num_clients_in_agg]
                    [sample] = random.sample(set_to_sample, 1)
                    head_output[sample] = fusion_model([embeddings[j] for j in sample])
                outputs.append(head_output)
        else:
            outputs = [{clients_l: fusion_model([embeddings[j] for j in clients_l]) for clients_l in self.powerset if i in clients_l} for i, fusion_model in enumerate(self.fusion_models)]

        return outputs

class FusionModel(nn.Module):

    def __init__(self, cut_dim, num_classes, aggregation="mean", num_clients=None):
        super().__init__()
        self.aggregation = aggregation
        if aggregation == 'conc':
            assert num_clients is not None
        fusion_input_dim = cut_dim * num_clients if aggregation == "conc" else cut_dim
        self.classifier = nn.Linear(fusion_input_dim, num_classes)

    def forward(self, x):
        if self.aggregation == 'sum':
            x = torch.stack(x).sum(dim=0)
        elif self.aggregation == 'mean':
            x = torch.stack(x).mean(dim=0)
        elif self.aggregation == 'conc':
            x = torch.cat(x, dim=1)
        pooled_view = self.classifier(x)
        return pooled_view

class StatEmbed(nn.Module):
    
    def __init__(self, code_vocab_size, embed_size, latent_size):             
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.code_vocab_size = code_vocab_size
        
        self.codeEmbed = nn.Embedding(self.code_vocab_size, self.embed_size)
        self.fc = nn.Linear(self.embed_size * self.code_vocab_size, self.latent_size, bias=True)
        
    def forward(self, code):
        # Create ids tensor of shape [sequence_length]
        ids = torch.arange(0, code.shape[1], device=code.device).long()

        # Get the embedded representation for each id
        code_embedded = self.codeEmbed(ids).unsqueeze(0).repeat(code.shape[0], 1, 1)

        # Perform element-wise multiplication
        code_embedded = code.unsqueeze(2) * code_embedded

        # Reshape the embedded tensor to [batch_size, embed_size * vocab_size]
        code_embedded = code_embedded.view(code_embedded.size(0), -1)

        # Pass through fully connected layer
        code_embedded = self.fc(code_embedded)
        
        return code_embedded
    
class CodeEmbed(nn.Module):

    def __init__(self, code_vocab_size, embed_size, latent_size):             
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.code_vocab_size = code_vocab_size
        
        # Define embedding layer and fully connected layer
        self.codeEmbed = nn.Embedding(self.code_vocab_size, self.embed_size)
        self.fc = nn.Linear(self.embed_size * self.code_vocab_size, self.latent_size, bias=True)
        
    def forward(self, code):
        # Create ids tensor of shape [sequence_length] and move to device
        ids = torch.arange(0, code.shape[2], device=code.device, dtype=torch.long)

        # Embed the ids and adjust dimensions to match the input
        code_embedded = self.codeEmbed(ids).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, sequence_length, embed_size]
        code_embedded = code_embedded.repeat(code.shape[0], code.shape[1], 1, 1)  # Repeat to match batch size and sequence length

        # Repeat the code tensor to match the embedding size and perform element-wise multiplication
        code = code.unsqueeze(3).repeat(1, 1, 1, code_embedded.shape[3])  # Shape: [batch_size, seq_len, vocab_size, embed_size]
        code_embedded = code * code_embedded  # Element-wise multiplication

        # Reshape the tensor to [batch_size, seq_len, embed_size * vocab_size]
        code_embedded = code_embedded.view(code_embedded.shape[0], code_embedded.shape[1], -1)

        # Pass through the fully connected layer
        code_embedded = self.fc(code_embedded)
        
        return code_embedded

class ValEmbed(nn.Module):

    def __init__(self, code_vocab_size, embed_size, latent_size):             
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.code_vocab_size = code_vocab_size
        
        # Build the model components
        self.codeEmbed = nn.BatchNorm1d(self.code_vocab_size)
        self.fc = nn.Linear(self.code_vocab_size, self.latent_size, bias=True)
        
    def forward(self, code):
        # Permute the input to match the expected shape [batch_size, vocab_size, sequence_length]
        code = code.permute(0, 2, 1).float()  # Convert to FloatTensor # code.permute(0, 2, 1).float() was code.permute(0, 2, 1).to(self.device).float()
        
        # Apply BatchNorm1d
        code_embedded = self.codeEmbed(code)
        
        # Permute back to the original shape [batch_size, sequence_length, vocab_size]
        code_embedded = code_embedded.permute(0, 2, 1)
        
        # Pass through the fully connected layer
        code_embedded = self.fc(code_embedded)
        
        return code_embedded

def split_into_chunks(lst, K):
    L = len(lst)
    base_size = L // K
    remainder = L % K  # The number of partitions that will have one extra element

    chunks = []
    start = 0
    for i in range(K):
        # Each chunk will have base_size elements, with an extra one for the first `remainder` chunks
        chunk_size = base_size + (1 if i < remainder else 0)
        chunks.append(lst[start:start + chunk_size])
        start += chunk_size
    
    return chunks

def drop_mask(plug_mask: torch.Tensor, p_drop: float) -> torch.Tensor:
    """
    Generate a new mask based on plug_mask and a dropout probability, ensuring the first entry is not dropped.

    Args:
    plug_mask (torch.Tensor): A 1D boolean tensor.
    p_drop (float): Probability of dropping an entry.

    Returns:
    torch.Tensor: A new 1D boolean tensor of the same shape as plug_mask.
    """
    random_probs = torch.rand_like(plug_mask, dtype=torch.float32)
    new_mask = plug_mask & (random_probs >= p_drop)
    new_mask[-1] = plug_mask[-1]  # the active party is not dropped
    return new_mask
