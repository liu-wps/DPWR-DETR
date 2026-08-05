"""Run standalone forward checks for the released DPWR-DETR core modules."""

import torch

from dpwr_detr import DGWAE, DIRDDecoderLayer, LUPH


def check_luph() -> None:
    module = LUPH(in_channels=16, hidden_dim=8).eval()
    with torch.inference_mode():
        d_route = module(torch.zeros(2, 16, 17, 19))
    assert d_route.shape == (2, 1, 17, 19)
    assert torch.isfinite(d_route).all()
    assert not any("query" in name.lower() for name, _ in module.named_parameters())


def check_dgwae() -> None:
    module = DGWAE(in_channels=(32, 16), out_channels=24, window_size=8, luph_hidden=8).eval()
    with torch.inference_mode():
        output, aux = module(
            [torch.zeros(2, 32, 17, 19), torch.zeros(2, 16, 17, 19)],
            return_aux=True,
        )
    assert output.shape == (2, 24, 17, 19)
    assert aux["d_route"].shape == (2, 1, 17, 19)
    assert aux["route_window"].shape[-2:] == (3, 3)


def check_dird() -> None:
    batch_size, num_queries, d_model = 2, 12, 32
    spatial_shapes = [(4, 4), (2, 2)]
    feature_length = sum(height * width for height, width in spatial_shapes)
    module = DIRDDecoderLayer(
        d_model=d_model,
        num_heads=4,
        d_ffn=64,
        num_levels=len(spatial_shapes),
        num_points=4,
    ).eval()
    with torch.inference_mode():
        output, aux = module(
            embedding=torch.zeros(batch_size, num_queries, d_model),
            reference_boxes=torch.sigmoid(torch.randn(batch_size, num_queries, 4)),
            features=torch.zeros(batch_size, feature_length, d_model),
            spatial_shapes=spatial_shapes,
            return_aux=True,
        )
    assert output.shape == (batch_size, num_queries, d_model)
    assert aux["q_first"].shape == output.shape
    assert aux["q_second"].shape == output.shape


if __name__ == "__main__":
    check_luph()
    check_dgwae()
    check_dird()
    print("Core module smoke tests passed.")
