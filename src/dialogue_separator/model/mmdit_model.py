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
        in_channels=4,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        max_seq_len=4096,
        mel_size=128,
        max_mel_len=4096,
        mel_hidden_size=512,
        ssl_size=1536,
        ssl_hidden_size=512,
        max_ssl_len=4096,
        heads=8,
    ):
        super(MMDiT, self).__init__()

        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.ssl_linear = nn.Sequential(
            nn.Linear(ssl_size, ssl_hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(ssl_hidden_size, ssl_hidden_size, bias=True),
        )
        self.mel_linear = nn.Sequential(
            nn.Linear(mel_size, mel_hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(mel_hidden_size, mel_hidden_size, bias=True),
        )
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size), requires_grad=False
        )
        self.ssl_pos_embed = nn.Parameter(
            torch.zeros(1, max_ssl_len, ssl_hidden_size), requires_grad=False
        )
        self.mel_pos_embed = nn.Parameter(
            torch.zeros(1, max_mel_len, mel_hidden_size), requires_grad=False
        )

        self.mmdit = mmdit.mmdit_generalized_pytorch.MMDiT(
            depth=depth,
            dim_modalities=(hidden_size, mel_hidden_size, ssl_hidden_size),
            dim_cond=hidden_size,
            qk_rmsnorm=True,
            flash_attn=True,
            heads=heads,
        )
        self.final_layer = nn.Linear(hidden_size, out_channels)
        self.max_seq_len = max_seq_len
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = positionalencoding1d(self.pos_embed.shape[-1], self.max_seq_len)
        self.pos_embed.data.copy_(pos_embed.float().unsqueeze(0))

        mel_pos_embed = positionalencoding1d(
            self.mel_pos_embed.shape[-1], self.mel_pos_embed.shape[1]
        )
        self.mel_pos_embed.data.copy_(mel_pos_embed.float().unsqueeze(0))
        ssl_pos_embed = positionalencoding1d(
            self.ssl_pos_embed.shape[-1], self.ssl_pos_embed.shape[1]
        )
        self.ssl_pos_embed.data.copy_(ssl_pos_embed.float().unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ssl_feature: torch.Tensor,
        mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if mel is None:
            raise ValueError("mel should not be None")
        if ssl_feature is None:
            raise ValueError("ssl_feature should not be None")
        x = self.x_embedder(x) + self.pos_embed[:, : x.shape[1], :]
        mel = self.mel_linear(mel) + self.mel_pos_embed[:, : mel.shape[1], :]
        ssl_feature = (
            self.ssl_linear(ssl_feature)
            + self.ssl_pos_embed[:, : ssl_feature.shape[1], :]
        )
        t = t * 1000

        t = self.t_embedder(t)  # (N, D)
        return self.final_layer(
            self.mmdit.forward(
                modality_tokens=(x, mel, ssl_feature),
                time_cond=t,
            )[0]
        )


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
