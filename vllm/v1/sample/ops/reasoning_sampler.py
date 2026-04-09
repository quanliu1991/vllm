# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

OPEN_THINK_TAG = 248068  # Qwen3.5 int(os.getenv("OPEN_THINK_TAG", "151667")) Qwen3
CLOSE_THINK_TAG = 248069  # Qwen3.5 int(os.getenv("CLOSE_THINK_TAG", "151668")) Qwen3


def apply_reasoning_stop_length(
    logits: torch.Tensor,
    output_token_ids: list[list[int]],
    max_thinking_tokens: int | list[int] = 4096,
    reasoning_open_think_in_prompt: list[bool] | None = None,
) -> torch.Tensor:
    """
    Enforce max thinking length by masking logits except for the close-think token.

    - **Qwen3 (older template)**: the model emits OPEN_THINK_TAG in the generated
      stream; we are inside thinking while OPEN is present and CLOSE has not
      appeared yet.
    - **Qwen3.5**: OPEN_THINK_TAG is in the prompt (position varies by template).
      ``reasoning_open_think_in_prompt[i]`` is True when the prompt contains
      OPEN_THINK_TAG but not CLOSE_THINK_TAG (thinking on). If both appear
      (empty thinking off), it is False.
    """
    stop_seq_id = []
    for i, one_output_token_ids in enumerate(output_token_ids):
        close_in_out = CLOSE_THINK_TAG in one_output_token_ids
        open_in_out = OPEN_THINK_TAG in one_output_token_ids

        if open_in_out:
            # Qwen3-style: open tag appears in model output.
            inside_thinking = not close_in_out
        elif (
            reasoning_open_think_in_prompt is not None
            and i < len(reasoning_open_think_in_prompt)
            and reasoning_open_think_in_prompt[i]
        ):
            # Qwen3.5-style: open tag only in prompt; generated stream is
            # reasoning until CLOSE appears.
            inside_thinking = not close_in_out
        else:
            inside_thinking = False

        if isinstance(max_thinking_tokens, list):
            limit = (
                max_thinking_tokens[i]
                if i < len(max_thinking_tokens)
                else 4096
            )
        else:
            limit = max_thinking_tokens
        if inside_thinking and len(one_output_token_ids) == limit:
            stop_seq_id.append(i)
    if stop_seq_id:
        logits.index_fill_(
            0,
            torch.tensor(
                stop_seq_id, device=logits.device, dtype=torch.int64
            ),
            float("-inf"),
        )
        rows = torch.tensor(stop_seq_id, device=logits.device)
        logits[rows, CLOSE_THINK_TAG] = 0.0

    return logits
