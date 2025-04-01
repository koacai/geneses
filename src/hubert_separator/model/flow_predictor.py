import math

import torch
from einops import pack
from torch import nn

from .transformer import BasicTransformerBlock


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x: torch.Tensor, scale: int = 1000) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super(TimestepEmbedding, self).__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super(Block1D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(nn.Module):
    def __init__(
        self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8
    ) -> None:
        super(ResnetBlock1D, self).__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super(Downsample1D, self).__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FlowPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int],
        dropout: float,
        attention_head_dim: int,
        n_blocks: int,
        act_fn: str,
    ) -> None:
        super(FlowPredictor, self).__init__()

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=time_embed_dim
        )

        self.down_blocks = nn.ModuleList([])
        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=attention_head_dim,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )
            self.down_blocks.append(
                nn.ModuleList([resnet, transformer_blocks, downsample])
            )

    def forward(
        self, x_merged: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        x_merged = x_merged.permute(0, 2, 1)
        x_t = x_t.permute(0, 2, 1)

        x = pack([x_merged, x_t], "b * t")[0]

        return x
