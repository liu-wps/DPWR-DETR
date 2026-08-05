"""LUPH and DGWAE core modules from DPWR-DETR."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    """Convolution followed by batch normalization and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DSConvNormAct(nn.Module):
    """Depthwise-separable convolution with batch normalization and GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteSEGate(nn.Module):
    """Lightweight channel gate used in the LUPH bottleneck."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


def _normalize_route_density(d_route: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    d_route = d_route.clamp(0.0, 1.0)
    if mode == "sigmoid_only":
        return d_route
    if mode == "max_norm":
        maximum = d_route.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
        return (d_route / maximum).clamp(0.0, 1.0)
    raise ValueError(f"Unsupported normalization mode: {mode}")


class LUPH(nn.Module):
    """Lite U-Net Prior Head that produces only the routing density map D_route."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 64,
        normalize_mode: str = "sigmoid_only",
        bottleneck_kernel: int = 5,
        use_local_residual: bool = True,
        local_res_gamma_init: float = 0.05,
        decoder_mode: str = "full",
    ):
        super().__init__()
        if bottleneck_kernel not in (5, 7):
            raise ValueError("bottleneck_kernel must be 5 or 7")
        if normalize_mode not in {"sigmoid_only", "max_norm"}:
            raise ValueError("normalize_mode must be 'sigmoid_only' or 'max_norm'")
        if decoder_mode not in {"full", "light"}:
            raise ValueError("decoder_mode must be 'full' or 'light'")

        self.normalize_mode = normalize_mode
        self.decoder_mode = decoder_mode
        self.use_local_residual = bool(use_local_residual)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.enc1 = DSConvNormAct(hidden_dim, hidden_dim)
        self.down1 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.enc2 = DSConvNormAct(hidden_dim, hidden_dim)
        self.down2 = nn.AvgPool2d(2, 2, ceil_mode=True)
        self.bottleneck = nn.Sequential(
            DSConvNormAct(hidden_dim, hidden_dim, bottleneck_kernel),
            LiteSEGate(hidden_dim),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            DSConvNormAct(hidden_dim, hidden_dim),
        )
        self.dec1_pre = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.dec1_refine = DSConvNormAct(hidden_dim, hidden_dim)
        self.local_residual = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            DSConvNormAct(hidden_dim, hidden_dim),
        )
        self.local_res_gamma = nn.Parameter(torch.tensor(float(local_res_gamma_init)))
        self.route_head = nn.Sequential(
            DSConvNormAct(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, shallow: torch.Tensor, out_size: tuple[int, int] | None = None) -> torch.Tensor:
        if shallow.ndim != 4:
            raise ValueError(f"LUPH expects [B,C,H,W], got {tuple(shallow.shape)}")

        e1 = self.enc1(self.stem(shallow))
        e2 = self.enc2(self.down1(e1))
        bottleneck = self.bottleneck(self.down2(e2))

        d2 = F.interpolate(bottleneck, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1_pre(torch.cat((d1, e1), dim=1))
        if self.decoder_mode == "full":
            d1 = self.dec1_refine(d1)

        prior_feature = d1
        if self.use_local_residual:
            gamma = self.local_res_gamma.to(dtype=d1.dtype)
            prior_feature = prior_feature + gamma * self.local_residual(e1)

        d_route = torch.sigmoid(self.route_head(prior_feature))
        if out_size is not None:
            d_route = F.interpolate(d_route, size=out_size, mode="bilinear", align_corners=False)
        return _normalize_route_density(d_route, self.normalize_mode)


class FourierFeatureMixer(nn.Module):
    """Frequency-domain feature enhancement used by the heavy route."""

    def __init__(self, channels: int):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, 1)
        self.branch2 = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x1 = self.branch1(x).float()
        x2 = self.branch2(x).float()
        x1_fft = torch.fft.fft2(x1, dim=(-2, -1), norm="backward")
        x2_fft = torch.fft.fft2(x2, dim=(-2, -1), norm="backward")
        amplitude = x1_fft.abs()
        mean = amplitude.mean(dim=(-2, -1), keepdim=True)
        std = amplitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
        frequency_gate = 1.0 + 0.5 * torch.tanh((amplitude - mean) / (std + 1e-6))
        enhanced = torch.fft.ifft2(frequency_gate * x2_fft, dim=(-2, -1), norm="backward").real
        return enhanced.to(dtype) * self.alpha.to(dtype) + x * self.beta.to(dtype)


class DGWAE(nn.Module):
    """Density-guided window-adaptive enhancement using LUPH's D_route."""

    def __init__(
        self,
        in_channels: tuple[int, int],
        out_channels: int | None = None,
        stride: int = 1,
        window_size: int = 8,
        density_mix: float = 0.5,
        route_threshold: float = 0.5,
        route_temperature: float = 0.3,
        hard_inference: bool = False,
        density_injection: float = 0.5,
        omega_alpha: float = 4.0,
        omega_beta: float = -2.0,
        density_pool_mode: str = "avgmax",
        luph_hidden: int = 64,
    ):
        super().__init__()
        if len(in_channels) != 2:
            raise ValueError("in_channels must be (main_channels, shallow_channels)")
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if density_pool_mode not in {"avgmax", "avgonly", "maxonly"}:
            raise ValueError("density_pool_mode must be avgmax, avgonly, or maxonly")

        main_channels, shallow_channels = map(int, in_channels)
        out_channels = main_channels if out_channels is None else int(out_channels)
        self.window_size = int(window_size)
        self.density_mix = float(density_mix)
        self.route_threshold = float(route_threshold)
        self.route_temperature = float(route_temperature)
        self.hard_inference = bool(hard_inference)
        self.density_injection = float(density_injection)
        self.omega_alpha = float(omega_alpha)
        self.omega_beta = float(omega_beta)
        self.density_pool_mode = density_pool_mode

        self.luph = LUPH(shallow_channels, hidden_dim=luph_hidden)
        if stride == 1:
            projection = ConvBNAct(main_channels, out_channels) if main_channels != out_channels else nn.Identity()
            self.heavy_branch = projection
            self.light_branch = (
                ConvBNAct(main_channels, out_channels) if main_channels != out_channels else nn.Identity()
            )
        else:
            self.heavy_branch = ConvBNAct(main_channels, out_channels, kernel_size=3, stride=2)
            self.light_branch = nn.Sequential(
                nn.AvgPool2d(2, 2, ceil_mode=True),
                ConvBNAct(main_channels, out_channels),
            )
        self.feature_mixer = FourierFeatureMixer(out_channels)

    def _pool_windows(self, d_route: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        _, _, height, width = d_route.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            d_route = F.pad(d_route, (0, pad_w, 0, pad_h), mode="replicate")
        avg_density = F.avg_pool2d(d_route, self.window_size, self.window_size)
        max_density = F.max_pool2d(d_route, self.window_size, self.window_size)
        if self.density_pool_mode == "avgonly":
            rho = avg_density
        elif self.density_pool_mode == "maxonly":
            rho = max_density
        else:
            rho = self.density_mix * avg_density + (1.0 - self.density_mix) * max_density
        return rho.clamp(0.0, 1.0), (height, width), (pad_h, pad_w)

    @staticmethod
    def _expand_window_map(
        window_map: torch.Tensor,
        input_size: tuple[int, int],
        padding: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        height, width = input_size
        pad_h, pad_w = padding
        expanded = F.interpolate(window_map, size=(height + pad_h, width + pad_w), mode="nearest")
        expanded = expanded[..., :height, :width]
        if expanded.shape[-2:] != output_size:
            expanded = F.interpolate(expanded, size=output_size, mode="nearest")
        return expanded

    def forward(
        self,
        features: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not isinstance(features, (tuple, list)) or len(features) != 2:
            raise ValueError("DGWAE expects [main_feature, shallow_feature]")
        main_feature, shallow_feature = features
        if main_feature.ndim != 4 or shallow_feature.ndim != 4:
            raise ValueError("DGWAE inputs must have shape [B,C,H,W]")
        if main_feature.shape[0] != shallow_feature.shape[0]:
            raise ValueError("DGWAE inputs must have the same batch size")

        input_size = main_feature.shape[-2:]
        d_route = self.luph(shallow_feature, out_size=input_size)
        rho_window, original_size, padding = self._pool_windows(d_route)
        route_window = torch.sigmoid(
            (rho_window - self.route_threshold) / max(self.route_temperature, 1e-6)
        )
        if not self.training and self.hard_inference:
            route_window = (route_window >= 0.5).to(route_window.dtype)
        omega_window = torch.sigmoid(self.omega_alpha * rho_window + self.omega_beta)

        heavy_base = self.heavy_branch(main_feature)
        output_size = heavy_base.shape[-2:]
        density_map = F.interpolate(d_route, size=output_size, mode="bilinear", align_corners=False)
        density_modulated = heavy_base * (1.0 + self.density_injection * density_map)
        enhanced = self.feature_mixer(density_modulated)

        omega_map = self._expand_window_map(omega_window, original_size, padding, output_size)
        heavy_feature = omega_map * enhanced + (1.0 - omega_map) * density_modulated
        light_feature = self.light_branch(main_feature)
        if light_feature.shape[-2:] != output_size:
            light_feature = F.interpolate(light_feature, size=output_size, mode="bilinear", align_corners=False)
        route_map = self._expand_window_map(route_window, original_size, padding, output_size)
        output = route_map * heavy_feature + (1.0 - route_map) * light_feature

        if not return_aux:
            return output
        return output, {
            "d_route": d_route,
            "rho_window": rho_window,
            "route_window": route_window,
            "omega_window": omega_window,
        }
