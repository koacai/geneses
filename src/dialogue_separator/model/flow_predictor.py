import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from einops import pack, rearrange

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

        self.res_conv = nn.Conv1d(dim, dim_out, 1)

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


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
    ):
        super(TimestepEmbedding, self).__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        return sample


class Upsample1D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = True,
        out_channels: int | None = None,
        name: str = "conv",
    ) -> None:
        super(Upsample1D, self).__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            assert self.conv is not None
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            assert self.conv is not None
            outputs = self.conv(outputs)

        return outputs


class MimiTokenEmbedding(nn.Module):
    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int) -> None:
        super(MimiTokenEmbedding, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Embedding(vocab_size, hidden_size) for _ in range(num_codebooks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_codebooks, length)
        """
        assert x.size(1) == len(self.linears)

        embeddings = []

        for i in range(x.size(1)):
            _embedding = self.linears[i](x[:, i, :])
            embeddings.append(_embedding)

        return torch.sum(torch.stack(embeddings), dim=0)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_codebooks: int,
        channels: tuple = (256, 256),
        dropout: float = 0.05,
        attention_head_dim: int = 64,
        n_blocks: int = 1,
        num_mid_blocks: int = 2,
        num_heads: int = 4,
        act_fn: str = "snake",
    ) -> None:
        super(Decoder, self).__init__()
        channels = tuple(channels)

        self.in_channels = 2 * hidden_size
        self.out_channels = hidden_size

        self.mimi_embedding = MimiTokenEmbedding(num_codebooks, vocab_size, hidden_size)

        self.time_embeddings = SinusoidalPosEmb(self.in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=self.in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = self.in_channels
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
                        attention_head_dim=num_heads,
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

        for i in range(num_mid_blocks):
            input_channel = channels[-1]

            resnet = ResnetBlock1D(
                dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim
            )

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=attention_head_dim,
                        attention_head_dim=num_heads,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=attention_head_dim,
                        attention_head_dim=num_heads,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.output_heads_1 = nn.ModuleList(
            [nn.Linear(self.out_channels, vocab_size) for _ in range(num_codebooks)]
        )
        self.output_heads_2 = nn.ModuleList(
            [nn.Linear(self.out_channels, vocab_size) for _ in range(num_codebooks)]
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x_t: torch.Tensor,
        mask: torch.Tensor,
        x_merged: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_t: (batch_size, num_codebooks, length, 2)
        """

        x_t_1 = x_t[:, :, :, 0]
        x_t_1 = self.mimi_embedding(x_t_1)
        x_t_2 = x_t[:, :, :, 1]
        x_t_2 = self.mimi_embedding(x_t_2)
        x_t = x_t_1 + x_t_2
        x_t = x_t.permute(0, 2, 1)

        x_merged = self.mimi_embedding(x_merged)
        x_merged = x_merged.permute(0, 2, 1)

        x_t = pack([x_t, x_merged], "b * t")[0]

        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:  # type: ignore
            mask_down = masks[-1]
            x_t = resnet(x_t, mask_down, t)
            x_t = rearrange(x_t, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x_t = transformer_block(
                    hidden_states=x_t,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x_t = rearrange(x_t, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x_t)
            x_t = downsample(x_t * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:  # type: ignore
            x_t = resnet(x_t, mask_mid, t)
            x_t = rearrange(x_t, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x_t = transformer_block(
                    hidden_states=x_t,
                    attention_mask=mask_mid,
                    timestep=t,
                )
            x_t = rearrange(x_t, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        mask_up = None
        for resnet, transformer_blocks, upsample in self.up_blocks:  # type: ignore
            mask_up = masks.pop()
            x_t = resnet(pack([x_t, hiddens.pop()], "b * t")[0], mask_up, t)
            x_t = rearrange(x_t, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x_t = transformer_block(
                    hidden_states=x_t,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x_t = rearrange(x_t, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x_t = upsample(x_t * mask_up)

        assert mask_up is not None
        x_t = self.final_block(x_t, mask_up)
        output = self.final_proj(x_t * mask_up)

        res = output * mask

        logits_1_list = []
        for layer in self.output_heads_1:
            _logit = layer(res.permute(0, 2, 1))
            logits_1_list.append(_logit)

        logits_1 = torch.stack(logits_1_list, dim=1)

        logits_2_list = []
        for layer in self.output_heads_2:
            _logit = layer(res.permute(0, 2, 1))
            logits_2_list.append(_logit)

        logits_2 = torch.stack(logits_2_list, dim=1)

        return torch.stack([logits_1, logits_2], dim=-1)
