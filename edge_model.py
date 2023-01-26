import os
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import Namespace
from typing import Dict, List, Union
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import degree
import pytorch_lightning as pl

from utils import (
    cal_auc_score, 
    cal_aupr_score, 
    cal_accuracy, 
    cal_cls_report, 
    classification_report,
    to_dense_adj,
)
from models.graph_base import (
    Data,
    Batch,
    Tensor,
    Adj,
    MLP,
    GCN,
)
from models.DOMINANT import DOMINANT_Base
from models.CONAD import CONAD_Base
from models.Anomaly_DAE import AnomalyDAE_Base
from models.SCAN import SCAN
from models.Dynamic_edge import DynamicEdge


class EdgeDetectionModel(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.args = hparams
        self.in_channels = self._get_hparam(hparams, 'feature_dim')
        self.embed_dim = 768

        # Logging
        self.start = datetime.now()

        # Logistics
        self.n_gpus = self._get_hparam(hparams, 'n_gpus', 1)
        self.checkpoint_dir = self._get_hparam(hparams, 'checkpoint_dir', '.')
        self.n_workers = self._get_hparam(hparams, 'n_workers', 1)
        self.event_only = self._get_hparam(hparams, 'event_only', False)

        # Training args
        self.lr = self._get_hparam(hparams, 'lr', 1e-3)
        self.weight_decay = self._get_hparam(hparams, 'weight_decay', 1e-5)
        self.train_batch_size = self._get_hparam(hparams, 'train_batch_size', 64)
        self.max_length = self._get_hparam(hparams, 'max_length', 1024)
        self.multi_granularity = self._get_hparam(hparams, 'multi_granularity', False)
        self.global_weight = self._get_hparam(hparams, 'global_weight', 0.5)

        # Model args
        model_kwargs = self._get_hparam(hparams, 'model_kwargs', dict())
        self.out_channels = model_kwargs.get('output_dim', 768)
        self.layers = model_kwargs.get('layers', 3)
        self.dropout = model_kwargs.get('dropout', 0.3)
        self.model_type = model_kwargs.get('model_type', 'dynamic')
        self.alpha = model_kwargs.get('alpha', 0.5)
        self.act = model_kwargs.get('act', F.relu)
        self.beta = model_kwargs.get('beta', 1.0)
        self.mu = model_kwargs.get('mu', 0.3)
        self.gamma = model_kwargs.get('gamma', 0.5)
        # Define edge score function parameters
        self.p_a = nn.Parameter(torch.DoubleTensor(self.embed_dim), requires_grad=True)
        self.p_b = nn.Parameter(torch.DoubleTensor(self.embed_dim), requires_grad=True)
        self.reset_parameters()

        # Models
        model_path = self._get_hparam(hparams, 'pretrained_model_path', 'facebook/bart-base')
        self.num_nodes = self._get_hparam(hparams, 'num_nodes')
        # Models
        if self.model_type == 'ae-dominant':
            self.model = DOMINANT_Base(
                in_dim=self.in_channels, 
                hid_dim=self.out_channels, 
                num_layers=self.layers, 
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'ae-anomalydae':
            self.num_nodes = self._get_hparam(hparams, 'num_nodes')
            self.model = AnomalyDAE_Base(
                in_node_dim=self.in_channels,
                in_num_dim=self.num_nodes,
                embed_dim=self.out_channels,
                out_dim=self.out_channels,
                dropout=self.dropout,
                act=self.act,
            )
            self.theta = model_kwargs.get('theta', 1.01)
            self.eta = model_kwargs.get('eta', 1.01)
        elif self.model_type == 'ae-conad':
            self.model = CONAD_Base(
                in_dim=self.in_channels,
                hid_dim=self.out_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
            self.r = model_kwargs.get('r', 0.2)
            self.m = model_kwargs.get('m', 50)
            self.k = model_kwargs.get('k', 50)
            self.f = model_kwargs.get('f', 10)
            self.eta = model_kwargs.get('eta', 0.5)
            margin = model_kwargs.get('margin', 0.5)
            self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)
        elif self.model_type == 'ae-gcnae':
            self.model = GCN(
                in_channels=self.in_channels,
                hidden_channels=self.out_channels,
                out_channels=self.in_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'ae-mlpae':
            self.model = MLP(
                in_channels=self.in_channels,
                hidden_channels=self.out_channels,
                out_channels=self.in_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        elif self.model_type == 'ae-scan':
            self.eps = model_kwargs.get('eps', 0.5)
            self.mu = model_kwargs.get('mu', 2)
            self.contamination = model_kwargs.get('contamination', 0.1)
            self.model = SCAN(
                eps=self.eps, 
                mu=self.mu, 
                contamination=self.contamination,
            )
        elif self.model_type == 'dynamic':
            self.model = DynamicEdge(
                model_path=model_path,
                in_channels=self.in_channels,
                num_nodes=self.num_nodes,
                out_channels=self.out_channels,
                num_layers=self.layers,
                dropout=self.dropout,
                act=self.act,
            )
        else:
            raise NotImplementedError('Model type {} not implemented'.format(self.model_type))
        
        # Logging
        print('Created {} module \n{} \nwith {:,} GPUs {:,} workers'.format(
            self.model.__class__.__name__, self.model, self.n_gpus, self.n_workers))
        # Save hyperparameters
        self.decision_scores = []
        # Loss
        self.mse_loss = MSELoss(reduction='none')
        # Save hyperparameters
        self.global_outputs = defaultdict(np.array)
        self.global_labels = defaultdict(np.array)
        self.train_dists = []
        self.train_avg = torch.normal(mean=0, std=1, size=(self.embed_dim,)) # E
        self.save_hyperparameters()
        
        
    def reset_parameters(self):
        p_a_ = self.p_a.unsqueeze(0)
        nn.init.xavier_uniform_(p_a_.data, gain=1.414)
        p_b_ = self.p_b.unsqueeze(0)
        nn.init.xavier_uniform_(p_b_.data, gain=1.414)
        
    @property
    def on_cuda(self):
        return next(self.parameters()).is_cuda

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default: bool = None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default

    def _sample_nodes(self, batch: Batch):
        perm = torch.randperm(batch.num_graphs)
        accum_nodes = 0
        data_list = []
        for graph_id in perm:
            data = batch.get_example(graph_id)
            if accum_nodes + data.num_nodes <= self.max_length:
                accum_nodes += data.num_nodes
                data_list.append(data)

        return batch.from_data_list(data_list)

    def score_func(self, hidden: Tensor, i: int, j: int, weight: float):
        # print('self.a: {}, self.b: {}'.format(self.p_a, self.p_b))
        s = self.p_a * hidden[i] + self.p_b * hidden[j]
        # print('s', s)
        s = F.dropout(s, self.dropout, training=self.training)
        # print('s', s)
        s_ = torch.norm(s, 2).pow(2)
        # print('s_', s_)
        score = weight * torch.sigmoid(self.beta * s_ - self.mu)
        # print('score', score)
        return score

    def neg_sampling(self, degrees: Tensor, i: int, j: int, s: Adj):
        if degrees.size()[0] == 2: 
            return None, None # no negative edge exists!
        # negative sampling
        prob_i = degrees[i]/(degrees[i] + degrees[j]) if degrees[i] + degrees[j] else 0
        if torch.rand(1).item() <= prob_i.item():
            if s[j].nonzero().size()[0] == s.size()[0]-1: # node j connect to all other nodes (except itself)
                return None, None # no negative edge exists!
            # replace node i
            i_prime = j
            while i_prime == j or s[i_prime, j] != 0:
                i_prime = torch.randint(s.size()[0], (1,)).item()
            return i_prime, j
        else:
            if s[i].nonzero().size()[0] == s.size()[0]-1: # node i connect to all other nodes (except itself)
                return None, None # no negative edge exists!
            # replace node j
            j_prime = i
            while j_prime == i or s[i, j_prime] != 0:
                j_prime = torch.randint(s.size()[0], (1,)).item()
            return i, j_prime

    def margin_loss(self, hidden: Tensor, G: Union[Data, Batch]):
        # hidden: |V| X E, G: |V| in |G|
        all_degrees = degree(G.edge_index[0], G.num_nodes)
        score = []
        loss = 0
        all_nodes = 0
        for k in range(G.num_graphs): 
            graph = G[k]
            # print('graph #{}: {}'.format(k, graph))
            graph_feature = hidden[all_nodes:all_nodes+graph.num_nodes]
            degrees = all_degrees[all_nodes:all_nodes+graph.num_nodes]
            s = G.s[all_nodes:all_nodes+graph.num_nodes, all_nodes:all_nodes+graph.num_nodes]
            all_nodes += graph.num_nodes
            for i, j in graph.edge_index.T.tolist():
                pos_score = self.score_func(graph_feature, i, j, s[i, j])
                # Negative sampling
                neg_score = float('-inf')
                search_time = 0
                while pos_score > neg_score and search_time < 10:
                    i_prime, j_prime = self.neg_sampling(degrees, i, j, s)
                    if (i_prime is not None) and (j_prime is not None):
                        # Found effective negative edge
                        neg_score = self.score_func(graph_feature, i_prime, j_prime, s[i, j])
                    search_time += 1
                
                # print('i {}, j {}，pos score {}'.format(i, j, pos_score))
                # print("i': {} j': {}, neg score: {}".format(i_prime, j_prime, neg_score))
                if pos_score <= neg_score:
                    edge_loss = F.relu(self.gamma + pos_score - neg_score)
                    # print('edge_loss', edge_loss)
                    loss += edge_loss
                    score.append(edge_loss.detach().cpu())

        if not score:
            score = torch.tensor([])
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            score = torch.stack(score)
        # print('loss', loss)
        return loss, score


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay, # l2 regularization
        )
        return optimizer

    def training_step(self, batch: Union[Data, Batch], batch_idx: int, split: str = 'train'): 
        # Sampling subgraph
        if batch.num_nodes > self.max_length:
            G = self._sample_nodes(batch)
        else:
            G = batch
            
        # G = train_batch
        # Generate adjacency matrix
        if not G.edge_index.shape[-1]: # empty edge index
            # print("Empty edge index !!!")
            G.s = torch.zeros((G.num_nodes, G.num_nodes))
            if self.on_cuda:
                G.s = G.s.cuda()
        else:
            G.s = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]

        # Automated balancing by std
        if self.alpha is None:
            self.alpha = torch.std(G.s).detach() / (torch.std(G.x).detach() + torch.std(G.s).detach())

        # Forward pass
        if self.model_type == 'dynamic':
            h = self.forward(
                x=G.x, 
                edge_index=G.edge_index, 
                batch=G.batch, 
                num_graphs=G.num_graphs, # for generating position embedding
            ) # |V| X E
        else:
            h = self.forward(
                x=G.x, 
                edge_index=G.edge_index,
            ) # |V| X E
        # print('output', h.shape)
        
        # Handling scores and loss
        labels = G.y
        # Calculate loss and save to dict
        loss, scores = self.margin_loss(h, G)
        
        if self.multi_granularity and self.model_type == 'dynamic':
            graph_feature = global_max_pool(h, G.batch)
            # print("graph feature", graph_feature.shape)
            # Handling average feature vector
            targets = self.train_avg.expand(graph_feature.shape[0], -1) # B X E
            if self.on_cuda:
                targets = targets.cuda()
            
            # Calculate loss and save to dict
            individual_loss = self.mse_loss(graph_feature, targets).sum(dim=-1) # B
            avg_loss = individual_loss.sum() # float
            loss = loss + self.global_weight * avg_loss # B
            if split == 'train':
                # Update train L2 distances
                self.train_dists.extend(individual_loss.detach().tolist())

        # Store training score distribution for analysis
        if split == 'train':
            self.decision_scores.extend(scores.detach().cpu().tolist())
        if split == 'test':
            labels = G.y[:scores.shape[0]] # needed when some of the nodes are cut
        # else:
        #     loss = torch.mean(scores)
        logging_dict = {'train_loss': loss.detach().item()}
        
        return {
            'loss': loss,
            'scores': scores,
            'preds': graph_feature, 
            'labels': labels,
            'log': logging_dict, # Tensorboard logging for training
            'progress_bar': logging_dict, # Progress bar logging for TQDM
        }

    def training_epoch_end(self, train_step_outputs: List[dict], split: str = 'train'):
        event_scores = torch.cat([instance['scores'].detach().cpu() for instance in train_step_outputs], dim=0) # N
        scores = event_scores.numpy() # N
        
        if split == 'train':
            # Update average train feature vector
            preds = [instance['preds'].detach().cpu() for instance in train_step_outputs] 
            self.train_avg = torch.cat(preds, dim=0).mean(dim=0) 
            # Update train dists and thresholds
            sorted_scores = sorted(scores)
            self.thre_max = max(scores)
            self.thre_mean = np.mean(scores)
            self.thre_top80 = sorted_scores[int(0.8*len(scores))]
            self.train_dists = []
        
            print("Epoch {} max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                self.current_epoch, 
                self.thre_max, 
                self.thre_top80, 
                self.thre_mean,
            ))
        elif split == 'val':
            # val_loss = sum(scores) / event_scores.shape[0] if event_scores.shape[0] else 0
            val_loss = sum(scores)
            print('Epoch {} val_loss: {:.4f}'.format(self.current_epoch, val_loss))
        else:
            event_labels = torch.cat([instance['labels'].detach().cpu() for instance in train_step_outputs], dim=0) # N
            scores = torch.cat([instance['scores'].detach().cpu() for instance in train_step_outputs], dim=0).numpy() # N
            labels = event_labels.numpy() # N
            if not hasattr(self, 'thre_max'):
                if self.decision_scores:
                    sorted_scores = sorted(self.decision_scores)
                    self.thre_max = max(self.decision_scores)
                    self.thre_mean = np.mean(self.decision_scores)
                    self.thre_top80 = sorted_scores[int(0.8*len(self.decision_scores))]
                else:
                    self.thre_max, self.thre_top80, self.thre_mean = 0.5, 0.5, 0.5
            print("Predicting {} test samples, {} ({:.2f}%) anomalies, using max thre {:.4f}, 80% thre {:.4f}, mean thre {:.4f}".format(
                len(labels),
                sum(labels),
                sum(labels)*100/len(labels),
                self.thre_max, 
                self.thre_top80, 
                self.thre_mean,
            ))
            # Calculating AUC
            auc_score = cal_auc_score(labels, scores)
            aupr_score = cal_aupr_score(labels, scores)
            # Threshold
            thre_dict = {
                'top80%': self.thre_top80, 
                'mean': self.thre_mean, 
                # 'max': self.thre_max,
            }
            pred_dict = defaultdict(np.array)
            for name, threshold in thre_dict.items():
                acc_score = cal_accuracy(labels, scores, threshold)
                pred_array, cls_report = cal_cls_report(labels, scores, threshold, output_dict=True)
                pred_results = {'AUC': [auc_score], 'AUPR': [aupr_score], 'ACC({})'.format(name): [acc_score]}
                stat_df = pd.DataFrame(pred_results)
                cls_df = pd.DataFrame(cls_report).transpose()
                pred_dict[name] = pred_array
                print(stat_df)
                print(cls_df)
                # Save predicting results (regarding each threshold)
                stat_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-results-{name}.csv'))
                cls_df.to_csv(os.path.join(self.checkpoint_dir, f'predict-cls-report-{name}.csv'))

            pred_dict['GT'] = labels
            pred_df = pd.DataFrame(pred_dict)
            pred_df.to_csv(os.path.join(self.checkpoint_dir, f'predictions.csv'))

        
    def validation_step(self, batch: Data, batch_idx: int, *args, **kwargs):
        loss_dict = self.training_step(batch, batch_idx, split='val')
        log_dict = loss_dict['log']
        log_dict['val_loss'] = log_dict.pop('train_loss')
        self.log("val_loss", log_dict['val_loss'], batch_size=loss_dict['scores'].size(0))
        return {
            'loss': loss_dict['loss'],
            'scores': loss_dict['scores'],
            'labels': loss_dict['labels'],
            'log': log_dict,
            'progress_bar': log_dict,
        }
    
    def validation_epoch_end(self, validation_step_outputs: List[dict]):
        self.training_epoch_end(validation_step_outputs, 'val')
    
    def test_step(self, batch: Data, batch_idx: int):
        loss_dict = self.training_step(batch, batch_idx, split='test')
        log_dict = loss_dict['log']
        log_dict['test_loss'] = log_dict.pop('train_loss')
        self.log("test_loss", log_dict['test_loss'], batch_size=loss_dict['scores'].size(0))
        return {
            'loss': loss_dict['loss'],
            'scores': loss_dict['scores'],
            'labels': loss_dict['labels'],
            'log': log_dict, # Tensorboard logging
            'progress_bar': log_dict, # Progress bar logging for TQDM
        }

    def test_epoch_end(self, test_step_outputs: List[dict]):
        self.training_epoch_end(test_step_outputs, 'test')