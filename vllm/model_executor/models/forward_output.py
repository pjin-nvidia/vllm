# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import torch

from vllm.model_executor.layers.fused_moe.router_output import FusedMoERouterOutput
from vllm.sequence import IntermediateTensors

HiddenStates = torch.Tensor | IntermediateTensors
AuxHiddenStates = list[torch.Tensor] | None
MoeHiddenStates = list[torch.Tensor] | None
MoeRouterOutputs = list[FusedMoERouterOutput] | None


class ModelForwardOutput(NamedTuple):
    """Structured output for model forward passes."""

    hidden_states: HiddenStates
    aux_hidden_states: AuxHiddenStates = None
    moe_input_hidden_states: MoeHiddenStates = None
    moe_output_hidden_states: MoeHiddenStates = None
    moe_router_outputs: MoeRouterOutputs = None
