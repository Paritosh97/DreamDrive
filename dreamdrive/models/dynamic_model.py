import torch
import torch.nn as nn
from torch import Tensor
import os
from dreamdrive.utils.loss import get_expon_lr_func
from dreamdrive.utils.general import searchForMaxIteration

class DynamicNetwork(torch.nn.Module):
    def __init__(self, in_ch, activation="sigmoid", bias=0.1):
        super(DynamicNetwork, self).__init__()
        self.in_ch = in_ch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            # torch.nn.Linear(in_ch, in_ch),
            # torch.nn.ReLU(),
            # torch.nn.Linear(in_ch, in_ch),
            # torch.nn.ReLU(),
            torch.nn.Linear(in_ch, 1),
        )
        self.activation = activation
        assert self.activation in ["sigmoid", "softplus", "no_act"]
        if self.activation == "softplus":
            self.activation_func = torch.nn.Softplus()
            self.bias = bias
        elif self.activation == "sigmoid":
            self.activation_func = torch.nn.Sigmoid()
        elif self.activation == "no_act":
            self.activation_func = torch.nn.Sequential()
        self.init()

    def init(self):
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if self.activation == "softplus":
            return self.bias + self.activation_func(self.net(features))  # bias for numerical stability
        elif self.activation == "sigmoid":
            return self.activation_func(self.net(features))
        elif self.activation == "no_act":
            return self.activation_func(self.net(features))

class DynamicModel:
    def __init__(self, in_ch=32):
        self.dynamic = DynamicNetwork(in_ch).cuda()
        self.optimizer = None

    def step(self, feats):
        return self.dynamic(feats)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.dynamic.parameters()),
             'lr': training_args.dynamic_net_lr,
             "name": "dynamic"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.dynamic_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_net_lr,
                                                       lr_final=training_args.dynamic_net_lr_final,
                                                       lr_delay_mult=training_args.dynamic_net_lr_delay_mult,
                                                       max_steps=training_args.dynamic_net_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "dynamic/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.dynamic.state_dict(), os.path.join(out_weights_path, 'dynamic.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "dynamic"))
        else:
            loaded_iter = iteration
        print("Loaded dynamic model from iteration {}".format(loaded_iter))
        weights_path = os.path.join(model_path, "dynamic/iteration_{}/dynamic.pth".format(loaded_iter))
        self.dynamic.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dynamic_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

class DynamicClusterModel(torch.nn.Module):
    def __init__(self, in_ch=32):
        super(DynamicClusterModel, self).__init__()
        self.dynamic = torch.nn.Sequential(
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
        ).cuda()
        self.cluster_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, 2),
        ).cuda()
        self.optimizer = None

    def step(self, feats, cluster_ids):
        """Cluster-based dynamic/static decomposition"""
        point_feats = self.dynamic(feats) # [N, ch]
        unique_clusters = torch.unique(cluster_ids)
        cluster_feats = []
        for cluster in unique_clusters:
            cluster_mask = (cluster_ids == cluster).squeeze(1)
            cluster_feat = point_feats[cluster_mask]
            cluster_feats.append(cluster_feat.mean(dim=0, keepdim=True))
        cluster_feats = torch.cat(cluster_feats, 0).contiguous() # [n_clusters, ch]
        cluster_feats = self.cluster_mlp(cluster_feats)
        cluster_probs = torch.nn.functional.gumbel_softmax(cluster_feats, dim=-1, hard=True) # [n_clusters, 2] [static, dynamic]
        dynamic_weights = torch.zeros((point_feats.shape[0], 1)).cuda()
        for cluster in unique_clusters:
            cluster_mask = (cluster_ids == cluster).squeeze(1)
            dynamic_weights[cluster_mask] = cluster_probs[cluster, 1]
        return dynamic_weights

    def train_setting(self, training_args):
        l = [
            {'params': list(self.parameters()),
             'lr': training_args.dynamic_net_lr,
             "name": "dynamic"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.dynamic_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_net_lr,
                                                       lr_final=training_args.dynamic_net_lr_final,
                                                       lr_delay_mult=training_args.dynamic_net_lr_delay_mult,
                                                       max_steps=training_args.dynamic_net_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "dynamic", "iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'dynamic.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "dynamic"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "dynamic/iteration_{}/dynamic.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dynamic_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

class DynamicClusterEmbeddingModel(torch.nn.Module):
    def __init__(self, in_ch=32, n_clusters=50):
        super(DynamicClusterEmbeddingModel, self).__init__()
        self.emb = nn.Parameter(torch.zeros((n_clusters, in_ch)), requires_grad=True)
        self.cluster_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, 1),
        )
        self.optimizer = None

    def step(self, feats, cluster_ids):
        """Cluster-based dynamic/static decomposition"""
        cluster_feats = self.emb
        cluster_feats = self.cluster_mlp(cluster_feats)
        cluster_probs = gumbel_sigmoid(cluster_feats, hard=False) # [n_clusters, 1] [static -> dynamic]
        dynamic_weights = torch.zeros((feats.shape[0], 1)).cuda()
        unique_clusters = torch.unique(cluster_ids)
        for cluster in unique_clusters:
            cluster_mask = (cluster_ids == cluster).squeeze(1)
            dynamic_weights[cluster_mask] = cluster_probs[cluster]
        return dynamic_weights

    def train_setting(self, training_args):
        l = [
            {'params': list(self.parameters()),
             'lr': training_args.dynamic_net_lr,
             "name": "dynamic"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.dynamic_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_net_lr,
                                                       lr_final=training_args.dynamic_net_lr_final,
                                                       lr_delay_mult=training_args.dynamic_net_lr_delay_mult,
                                                       max_steps=training_args.dynamic_net_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "dynamic", "iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'dynamic.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "dynamic"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "dynamic/iteration_{}/dynamic.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dynamic_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

def gumbel_sigmoid_noise(logits: torch.Tensor) -> torch.Tensor:
    eps = 3e-4 if logits.dtype == torch.float16 else 1e-10
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(eps, 1 - eps)

    noise = -(uniform[1].log() / uniform[0].log() + eps).log()
    return noise


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False) -> torch.Tensor:
    logits = logits + gumbel_sigmoid_noise(logits)
    res = torch.sigmoid(logits / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res

class DynamicPointModel(torch.nn.Module):
    def __init__(self, in_ch=32, n_points=50):
        super(DynamicPointModel, self).__init__()
        self.emb = nn.Parameter(torch.zeros((n_points, in_ch)), requires_grad=True)
        self.point_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, in_ch),
            torch.nn.ReLU(),
            torch.nn.Linear(in_ch, 1),
        )
        self.activation = torch.nn.Sigmoid()
        self.optimizer = None

    def step(self, feats):
        """Cluster-based dynamic/static decomposition"""
        point_feats = self.emb
        point_feats = self.point_mlp(point_feats)
        point_probs = self.activation(point_feats) # [n_points, 1] [static -> dynamic]
        return point_probs

    def train_setting(self, training_args):
        l = [
            {'params': list(self.parameters()),
             'lr': training_args.dynamic_net_lr,
             "name": "dynamic"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.dynamic_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_net_lr,
                                                       lr_final=training_args.dynamic_net_lr_final,
                                                       lr_delay_mult=training_args.dynamic_net_lr_delay_mult,
                                                       max_steps=training_args.dynamic_net_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "dynamic", "iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out_weights_path, 'dynamic.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "dynamic"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "dynamic/iteration_{}/dynamic.pth".format(loaded_iter))
        self.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "dynamic":
                lr = self.dynamic_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
