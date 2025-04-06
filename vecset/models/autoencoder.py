# Copyright (c) 2025, Biao Zhang.

import torch
import torch.nn as nn

from torch_cluster import fps

from einops import rearrange, repeat

import math

from .utils import PreNorm, Attention, FeedForward, subsample
from .utils import PointEmbed
from .bottleneck import Bottleneck, KLBottleneck, NormalizedBottleneck

class VecSetAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=768,
        output_dim=1,
        num_inputs=2048,
        num_latents=1280,
        latent_dim=16,
        dim_head=64,
        query_type='point',
        bottleneck=None,
        bottleneck_args={},
    ):
        super().__init__()

        queries_dim = dim
        
        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents
        
        self.query_type = query_type
        if query_type == 'point':
            pass
        elif query_type == 'learnable':
            self.latents = nn.Embedding(num_latents, dim)
        else:
            raise NotImplementedError(f'Query type {query_type} not implemented')

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = dim // dim_head, dim_head = dim_head)),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)
        
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = dim // dim_head, dim_head = dim_head)),
                PreNorm(dim, FeedForward(dim))
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = dim // dim_head, dim_head = dim_head))

        self.to_outputs = nn.Sequential(
            nn.LayerNorm(queries_dim),
            nn.Linear(queries_dim, output_dim)
        )
        
        nn.init.zeros_(self.to_outputs[1].weight)
        nn.init.zeros_(self.to_outputs[1].bias)
        
        self.bottleneck = bottleneck(**bottleneck_args)


    def encode(self, pc):
        B, N, _ = pc.shape
        assert N == self.num_inputs
        
        if self.query_type == 'point':
            sampled_pc = subsample(pc, N, self.num_latents)
            x = self.point_embed(sampled_pc)
        elif self.query_type == 'learnable':
            x = repeat(self.latents.weight, 'n d -> b n d', b = B)
            
        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context = pc_embeddings, mask = None) + x
        x = cross_ff(x) + x

        bottleneck = self.bottleneck.pre(x)
        return bottleneck

    def learn(self, x):

        x = self.bottleneck.post(x)
        
        if self.query_type == 'learnable':
            x = x + self.latents.weight[None]

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        
        return x
    
    def decode(self, x, queries):
        
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        bottleneck = self.encode(pc)
        x = self.learn(bottleneck['x'])

        if queries.shape[1] > 100000:
            N = 100000
            os = []
            for block_idx in range(math.ceil(queries.shape[1] / N)):
                o = self.decode(x, queries[:, block_idx*N:(block_idx+1)*N, :]).squeeze(-1)
                os.append(o)
            o = torch.cat(os, dim=1)
        else:
            o = self.decode(x, queries).squeeze(-1)

        return {'o': o, **bottleneck}

def create_autoencoder(depth=24, dim=512, M=512, N=2048, query_type='point', bottleneck=None, bottleneck_args={}):
    model = VecSetAutoEncoder(
        depth=depth, dim=dim, output_dim=1, num_inputs=N, 
        num_latents=M, query_type=query_type, 
        bottleneck=bottleneck, 
        bottleneck_args=bottleneck_args,
    )
    return model

def learnable_vec1024x16_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='learnable', 
        bottleneck=NormalizedBottleneck, 
        bottleneck_args={'dim': 1024, 'latent_dim': 16},
    )

def learnable_vec1024x32_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='learnable', 
        bottleneck=NormalizedBottleneck, 
        bottleneck_args={'dim': 1024, 'latent_dim': 32},
    )
    
def point_vec1024x16_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='point', 
        bottleneck=NormalizedBottleneck, 
        bottleneck_args={'dim': 1024, 'latent_dim': 16},
    )

def point_vec1024x32_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='point', 
        bottleneck=NormalizedBottleneck, 
        bottleneck_args={'dim': 1024, 'latent_dim': 32},
    )
    
def learnable_vec1024_dim1024_depth24(pc_size=8192):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='learnable', 
        bottleneck=Bottleneck, 
        bottleneck_args={},
    )