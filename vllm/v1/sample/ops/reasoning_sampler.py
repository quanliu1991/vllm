# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

OPEN_THINK_TAG = int(os.getenv("OPEN_THINK_TAG", "151667"))
CLOSE_THINK_TAG = int(os.getenv("CLOSE_THINK_TAG", "151668"))

def apply_reasoning_stop_length(
    logits: torch.Tensor,
    output_token_ids: list[list[int]],
    max_thinking_tokens: int = 4096
) -> torch.Tensor:
    """
    Applies max_thinking_token to the logits.
    """
    stop_seq_id = []
    for i, one_output_token_ids in enumerate(output_token_ids):
        if (OPEN_THINK_TAG in one_output_token_ids
                and CLOSE_THINK_TAG not in one_output_token_ids):
            if len(one_output_token_ids) == max_thinking_tokens:
                stop_seq_id.append(i)
    if stop_seq_id:
        logits.index_fill_(0, torch.tensor(stop_seq_id,
                                           device=logits.device,
                                           dtype=torch.int64), float('-inf'))
        rows = torch.tensor(stop_seq_id, device=logits.device)
        logits[rows, CLOSE_THINK_TAG] = 0.0

    return logits


