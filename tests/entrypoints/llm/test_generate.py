# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "distilbert/distilgpt2"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    """We can run both engines for this test."""
    pass


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME,
              max_num_batched_tokens=4096,
              tensor_parallel_size=1,
              gpu_memory_utilization=0.10,
              enforce_eager=True)

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup_dist_env_and_memory()


def assert_outputs_equal(o1: list[RequestOutput], o2: list[RequestOutput]):
    assert [o.outputs for o in o1] == [o.outputs for o in o2]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize('prompt_token_ids', TOKEN_IDS)
def test_v1_v2_api_consistency_single_prompt_tokens(llm: LLM,
                                                    prompt_token_ids):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
        v1_output = llm.generate(prompt_token_ids=prompt_token_ids,
                                 sampling_params=sampling_params)

    v2_output = llm.generate({"prompt_token_ids": prompt_token_ids},
                             sampling_params=sampling_params)
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_v1_v2_api_consistency_multi_prompt_tokens(llm: LLM):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0)

    with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
        v1_output = llm.generate(prompt_token_ids=TOKEN_IDS,
                                 sampling_params=sampling_params)

    v2_output = llm.generate(
        [{
            "prompt_token_ids": p
        } for p in TOKEN_IDS],
        sampling_params=sampling_params,
    )
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_multiple_sampling_params(llm: LLM):
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(PROMPTS, sampling_params=None)
    assert len(PROMPTS) == len(outputs)


def test_max_model_len():
    max_model_len = 20
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.10,
        enforce_eager=True,  # reduce test time
    )
    sampling_params = SamplingParams(max_tokens=max_model_len + 10)
    outputs = llm.generate(PROMPTS, sampling_params)
    for output in outputs:
        num_total_tokens = len(output.prompt_token_ids) + len(
            output.outputs[0].token_ids)
        # Total tokens must not exceed max_model_len.
        # It can be less if generation finishes due to other reasons (e.g., EOS)
        # before reaching the absolute model length limit.
        assert num_total_tokens <= max_model_len
