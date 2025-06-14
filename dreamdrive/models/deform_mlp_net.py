import torch
import torch.nn as nn
import torch.nn.functional as F

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, use_semantic_feats=False):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)

        self.input_ch = xyz_input_ch + time_input_ch

        self.use_semantic_feats = use_semantic_feats
        if use_semantic_feats: # add directly to inputs
            self.input_ch += 32

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

        self.init()

    def init(self):
        nn.init.zeros_(self.gaussian_warp.weight)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_rotation.weight)
        nn.init.zeros_(self.gaussian_rotation.bias)
        nn.init.zeros_(self.gaussian_scaling.weight)
        nn.init.zeros_(self.gaussian_scaling.bias)

    def forward(self, x, t, feats=None):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        if self.use_semantic_feats:
            h = torch.cat([x_emb, t_emb, feats], dim=-1)
        else:
            h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                if self.use_semantic_feats:
                    h = torch.cat([h, x_emb, t_emb, feats], dim=-1)
                else:
                    h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

class DeformOpacityNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, multires=10, use_semantic_feats=False):
        super(DeformOpacityNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.t_multires = 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, 3)

        self.input_ch = xyz_input_ch + time_input_ch

        self.use_semantic_feats = use_semantic_feats
        if use_semantic_feats: # add directly to inputs
            self.input_ch += 32

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)
        self.gaussian_opacity = nn.Linear(W, 1)

        self.init()

    def init(self):
        nn.init.zeros_(self.gaussian_warp.weight)
        nn.init.zeros_(self.gaussian_warp.bias)
        nn.init.zeros_(self.gaussian_rotation.weight)
        nn.init.zeros_(self.gaussian_rotation.bias)
        nn.init.zeros_(self.gaussian_scaling.weight)
        nn.init.zeros_(self.gaussian_scaling.bias)
        nn.init.zeros_(self.gaussian_opacity.weight)
        nn.init.zeros_(self.gaussian_opacity.bias)

    def forward(self, x, t, feats=None):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        if self.use_semantic_feats:
            h = torch.cat([x_emb, t_emb, feats], dim=-1)
        else:
            h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                if self.use_semantic_feats:
                    h = torch.cat([h, x_emb, t_emb, feats], dim=-1)
                else:
                    h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)
        opacity = self.gaussian_opacity(h)
        return d_xyz, rotation, scaling, opacity
