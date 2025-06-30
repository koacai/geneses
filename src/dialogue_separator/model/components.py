# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import mmdit
import mmdit.mmdit_generalized_pytorch
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MMDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_ssl_channels: int,
        out_channels: int,
        hidden_size: int,
        max_ssl_seq_len: int,
        max_seq_len: int,
        depth: int,
        heads: int,
    ) -> None:
        super(MMDiT, self).__init__()

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.x_embedder_merged = nn.Linear(in_ssl_channels, hidden_size, bias=True)
        self.x_embedder_1 = nn.Linear(in_channels, hidden_size, bias=True)
        self.x_embedder_2 = nn.Linear(in_channels, hidden_size, bias=True)

        self.x_pos_embed_merged = nn.Parameter(
            torch.zeros(1, max_ssl_seq_len, hidden_size), requires_grad=False
        )
        self.x_pos_embed_1 = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size), requires_grad=False
        )
        self.x_pos_embed_2 = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size), requires_grad=False
        )

        self.mmdit = mmdit.mmdit_generalized_pytorch.MMDiT(
            depth=depth,
            dim_modalities=(hidden_size, hidden_size, hidden_size),
            dim_cond=hidden_size,
            qk_rmsnorm=True,
            flash_attn=True,
            heads=heads,
        )

        self.final_layer_1 = nn.Linear(hidden_size, out_channels)
        self.final_layer_2 = nn.Linear(hidden_size, out_channels)

        self.max_seq_len = max_ssl_seq_len

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        x_pos_embed_merged = positionalencoding1d(
            self.x_pos_embed_merged.shape[-1], self.max_seq_len
        )
        self.x_pos_embed_merged.data.copy_(x_pos_embed_merged.float().unsqueeze(0))

        x_pos_embed_1 = positionalencoding1d(
            self.x_pos_embed_1.shape[-1], self.x_pos_embed_1.shape[1]
        )
        self.x_pos_embed_1.data.copy_(x_pos_embed_1.float().unsqueeze(0))

        x_post_embed_2 = positionalencoding1d(
            self.x_pos_embed_2.shape[-1], self.x_pos_embed_2.shape[1]
        )
        self.x_pos_embed_2.data.copy_(x_post_embed_2.float().unsqueeze(0))

    def forward(
        self,
        x_merged: torch.Tensor,
        t: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_merged = (
            self.x_embedder_merged(x_merged)
            + self.x_pos_embed_merged[:, : x_merged.shape[1], :]
        )
        x_1 = self.x_embedder_1(x_1) + self.x_pos_embed_1[:, : x_1.shape[1], :]
        x_2 = self.x_embedder_2(x_2) + self.x_pos_embed_2[:, : x_2.shape[1], :]
        t = t * 1000

        t = self.t_embedder(t)  # (N, D)

        out = self.mmdit.forward(
            modality_tokens=(x_merged, x_1, x_2),
            time_cond=t,
        )
        res_1 = self.final_layer_1(out[1])
        res_2 = self.final_layer_2(out[2])

        return res_1, res_2


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(
                d_model
            )
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe
