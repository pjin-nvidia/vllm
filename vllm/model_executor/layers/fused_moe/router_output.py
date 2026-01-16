# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch


class FusedMoERouterOutput(NamedTuple):
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


class FusedMoEForwardOutput(NamedTuple):
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    router_output: FusedMoERouterOutput
