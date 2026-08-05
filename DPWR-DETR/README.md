# DPWR-DETR Core Modules

Partial reference implementation of DPWR-DETR for dense small-object detection in aerial and remote-sensing images.

## Code Availability

This repository contains a partial code release for peer review. It provides the core implementations of:

- `LUPH`: produces only the routing density prior `D_route`
- `DGWAE`: performs density-guided adaptive window routing
- `DIRDDecoderLayer`: performs dual-inquiry residual decoder refinement

The complete detector assembly, training and evaluation pipelines, experiment configurations, checkpoints, datasets, and result files are not included in this release. **The complete code will be made publicly available after the paper is accepted.**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch

from dpwr_detr import DGWAE, DIRDDecoderLayer, LUPH

# LUPH returns one tensor: D_route.
luph = LUPH(in_channels=64, hidden_dim=32)
d_route = luph(torch.randn(1, 64, 32, 32))

# DGWAE consumes a main feature and a shallow feature.
dgwae = DGWAE(in_channels=(128, 64), out_channels=128)
feature = dgwae([torch.randn(1, 128, 32, 32), torch.randn(1, 64, 32, 32)])

# DIRD is intended to replace selected tail layers of a deformable decoder.
dird = DIRDDecoderLayer(d_model=256, num_heads=8, num_levels=3)
```

Run the standalone structure checks with:

```bash
python smoke_test.py
```

## Scope

The released modules are provided for architectural reference. This partial repository is not presented as a standalone reproduction of all quantitative results reported in the paper.

## Citation and License

See [CITATION.cff](CITATION.cff) for citation metadata. This partial release is distributed under the AGPL-3.0 license; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
