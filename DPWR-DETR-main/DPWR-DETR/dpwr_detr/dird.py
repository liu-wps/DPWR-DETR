"""DIRD decoder layer from DPWR-DETR."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_scale_deformable_attention(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor | list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch multi-scale deformable attention sampling."""
    batch_size, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    shapes = [(int(height), int(width)) for height, width in spatial_shapes]
    value_levels = value.split([height * width for height, width in shapes], dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0

    sampled_levels = []
    for level, (height, width) in enumerate(shapes):
        value_level = value_levels[level].flatten(2).transpose(1, 2)
        value_level = value_level.reshape(batch_size * num_heads, head_dim, height, width)
        sampling_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_levels.append(
            F.grid_sample(
                value_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (torch.stack(sampled_levels, dim=-2).flatten(-2) * weights).sum(-1)
    output = output.view(batch_size, num_heads * head_dim, num_queries)
    return output.transpose(1, 2).contiguous()


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention used by each DIRD inquiry branch."""

    def __init__(self, d_model: int = 256, num_levels: int = 4, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_levels = int(num_levels)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.sampling_offsets = nn.Linear(d_model, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(d_model, num_heads * num_levels * num_points)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        angles = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid = torch.stack((angles.cos(), angles.sin()), dim=-1)
        grid = grid / grid.abs().amax(dim=-1, keepdim=True)
        grid = grid.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for point in range(self.num_points):
            grid[:, :, point] *= point + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.reshape(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.constant_(self.value_projection.bias, 0.0)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor | list[tuple[int, int]],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_queries = query.shape[:2]
        value_length = value.shape[1]
        shapes = [(int(height), int(width)) for height, width in spatial_shapes]
        if sum(height * width for height, width in shapes) != value_length:
            raise ValueError("spatial_shapes do not match the flattened feature length")

        value = self.value_projection(value)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
        value = value.view(batch_size, value_length, self.num_heads, self.d_model // self.num_heads)

        offsets = self.sampling_offsets(query).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        weights = F.softmax(weights, dim=-1).view(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_boxes.shape[-1] == 2:
            normalizer = torch.as_tensor(shapes, dtype=query.dtype, device=query.device).flip(-1)
            sampling_locations = reference_boxes[:, :, None, :, None] + (
                offsets / normalizer[None, None, None, :, None]
            )
        elif reference_boxes.shape[-1] == 4:
            scaled_offsets = offsets / self.num_points * reference_boxes[:, :, None, :, None, 2:] * 0.5
            sampling_locations = reference_boxes[:, :, None, :, None, :2] + scaled_offsets
        else:
            raise ValueError("reference_boxes must end with 2 or 4 coordinates")

        output = multi_scale_deformable_attention(value, shapes, sampling_locations, weights)
        return self.output_projection(output)


class CrossAttentionFFN(nn.Module):
    """Independent cross-attention and feed-forward path for one inquiry."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float,
        activation: nn.Module,
        num_levels: int,
        num_points: int,
    ):
        super().__init__()
        self.cross_attention = MSDeformableAttention(d_model, num_levels, num_heads, num_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        inquiry: torch.Tensor,
        reference_boxes: torch.Tensor,
        features: torch.Tensor,
        spatial_shapes: torch.Tensor | list[tuple[int, int]],
        padding_mask: torch.Tensor | None,
        query_position: torch.Tensor | None,
    ) -> torch.Tensor:
        query = inquiry if query_position is None else inquiry + query_position
        update = self.cross_attention(
            query,
            reference_boxes.unsqueeze(2),
            features,
            spatial_shapes,
            padding_mask,
        )
        inquiry = self.norm1(inquiry + self.dropout1(update))
        update = self.linear2(self.dropout2(self.activation(self.linear1(inquiry))))
        return self.norm2(inquiry + self.dropout3(update))


class DIRDDecoderLayer(nn.Module):
    """Dual-Inquiry Residual Decoder layer with one base and two inquiry paths."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        num_levels: int = 4,
        num_points: int = 4,
        gamma_init: float = 0.1,
        enabled: bool = True,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.self_dropout = nn.Dropout(dropout)
        self.self_norm = nn.LayerNorm(d_model)

        def make_path() -> CrossAttentionFFN:
            return CrossAttentionFFN(
                d_model,
                num_heads,
                d_ffn,
                dropout,
                nn.ReLU(),
                num_levels,
                num_points,
            )

        self.base_path = make_path()
        self.inquiry_paths = nn.ModuleList((make_path(), make_path()))
        self.inquiry_fusion = nn.Linear(d_model * 2, d_model)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.xavier_uniform_(self.inquiry_fusion.weight, gain=0.1)
        nn.init.constant_(self.inquiry_fusion.bias, 0.0)

    def forward(
        self,
        embedding: torch.Tensor,
        reference_boxes: torch.Tensor,
        features: torch.Tensor,
        spatial_shapes: torch.Tensor | list[tuple[int, int]],
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_position: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = embedding if query_position is None else embedding + query_position
        self_update = self.self_attention(
            query.transpose(0, 1),
            query.transpose(0, 1),
            embedding.transpose(0, 1),
            attn_mask=attention_mask,
        )[0].transpose(0, 1)
        shared = self.self_norm(embedding + self.self_dropout(self_update))

        q_base = self.base_path(
            shared, reference_boxes, features, spatial_shapes, padding_mask, query_position
        )
        if not self.enabled:
            return (q_base, {"q_base": q_base}) if return_aux else q_base

        q_first = self.inquiry_paths[0](
            shared, reference_boxes, features, spatial_shapes, padding_mask, query_position
        )
        q_second = self.inquiry_paths[1](
            shared, reference_boxes, features, spatial_shapes, padding_mask, query_position
        )
        q_multi = self.inquiry_fusion(torch.cat((q_first, q_second), dim=-1))
        gamma = self.gamma.to(dtype=q_base.dtype)
        q_output = q_base + gamma * (q_multi - q_base)

        if not return_aux:
            return q_output
        return q_output, {
            "q_base": q_base,
            "q_first": q_first,
            "q_second": q_second,
            "q_multi": q_multi,
            "gamma": self.gamma,
        }
