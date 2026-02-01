"""
Test pause and continue generation with inference engine client HTTP endpoint.

uv run --isolated --extra dev --extra sglang pytest tests/gpu/gpu_ci/test_pause_and_continue_generation.py -m "sglang"
"""

import pytest
import asyncio
from tests.gpu.gpu_ci.test_inference_engine_client_http_endpoint import get_test_actor_config
from tests.gpu.utils import init_inference_engines, get_test_prompts
from skyrl_train.inference_engines.base import ConversationType
from transformers import AutoTokenizer
from typing import List
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    serve,
    wait_for_server_ready,
    shutdown_server,
)
import threading
import requests

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1  # SGLang currently only supports TP=1
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"


@pytest.mark.sglang
def test_continue_generation_sglang_engine_generate(ray_init_fixture):
    """
    Launch 6 concurrent single-request generate() calls against two engines with SGLang.
    Ignore EOS and request a long generation (2048 tokens).
    Pause and then resume generation twice mid-flight. Expect each request to finish with reason `length`
    and have exactly `max_tokens` completion tokens.
    """
    num_engines = 2
    num_requests = 6
    max_tokens = 2048

    # 1. Build engines (no HTTP server needed for generate())
    cfg = get_test_actor_config(num_inference_engines=num_engines, model=MODEL)
    cfg.trainer.placement.colocate_all = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    sampling_params = {
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        # SGLang uses return_logprob instead of logprobs
        "return_logprob": True,
    }
    client, _ = init_inference_engines(
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend="sglang",
        model=MODEL,
        num_inference_engines=cfg.generator.num_inference_engines,
        sleep_level=1,
    )

    # 2. Prepare a single ConversationType prompt; each generate() call will be single-request
    messages: List[ConversationType] = get_test_prompts(MODEL, num_samples=1)[0]

    # 3. Fire 6 concurrent client.generate() single-request calls, then pause/resume mid-flight
    async def run_requests_then_pause():
        async def one_req(i: int):
            engine_input = {
                "prompts": [messages],  # single request path
                "prompt_token_ids": None,
                "sampling_params": dict(sampling_params),
                "session_ids": [i],
            }
            return await client.generate(engine_input)

        tasks = [asyncio.create_task(one_req(i)) for i in range(num_requests)]
        # Let requests start and enqueue
        await asyncio.sleep(1)
        # Pause then resume while requests are in-flight
        await client.pause_generation()
        await client.resume_generation()
        # Run for another two seconds, then pause and resume again
        await asyncio.sleep(2)
        await client.pause_generation()
        await client.resume_generation()
        return await asyncio.gather(*tasks)

    outputs = asyncio.run(run_requests_then_pause())

    # 4. Validate each output: stop_reason is "length" and tokens == max_tokens
    assert len(outputs) == num_requests, f"Expected {num_requests} outputs, got {len(outputs)}"
    for i, out in enumerate(outputs):
        # InferenceEngineOutput shape checks
        assert "responses" in out and "response_ids" in out and "stop_reasons" in out
        assert len(out["responses"]) == 1 and len(out["response_ids"]) == 1 and len(out["stop_reasons"]) == 1
        assert out["stop_reasons"][0] == "length", f"Request {i} stop_reason is not 'length': {out['stop_reasons'][0]}"
        # Check completion tokens via response_ids
        token_ids = out["response_ids"][0]
        assert (
            len(token_ids) == sampling_params["max_tokens"]
        ), f"Request {i} expected {sampling_params['max_tokens']} tokens, got {len(token_ids)}"
        # Check response_logprobs length if present
        if "response_logprobs" in out and out["response_logprobs"] is not None:
            assert (
                len(out["response_logprobs"][0]) == sampling_params["max_tokens"]
            ), f"Request {i} expected {sampling_params['max_tokens']} logprobs, got {len(out['response_logprobs'][0])}"
        # Check string output decodes correctly
        assert out["responses"][0] == client.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Print a preview to aid debugging
        print(f"Output first 1500 chars: {out['responses'][0][:1500]}...")


@pytest.mark.sglang
def test_continue_generation_sglang_engine_with_logprobs(ray_init_fixture):
    """
    Launch 6 concurrent single-request generate() calls against two engines with SGLang.
    Ignore EOS and request a long generation (2048 tokens).
    Pause and then resume generation twice mid-flight. Expect each request to finish with reason `length`
    and have exactly `max_tokens` completion tokens.
    """
    num_engines = 2
    num_requests = 6
    max_tokens = 2048

    # 1. Build engines (no HTTP server needed for generate())
    cfg = get_test_actor_config(num_inference_engines=num_engines, model=MODEL)
    cfg.trainer.placement.colocate_all = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    sampling_params = {
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        # SGLang uses return_logprob instead of logprobs
        "return_logprob": True,
    }
    client, _ = init_inference_engines(
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend="sglang",
        model=MODEL,
        num_inference_engines=cfg.generator.num_inference_engines,
        sleep_level=1,
    )

    # 2. Prepare a single ConversationType prompt; each generate() call will be single-request
    messages: List[ConversationType] = get_test_prompts(MODEL, num_samples=1)[0]

    # 3. Fire 6 concurrent client.generate() single-request calls, then pause/resume mid-flight
    async def run_requests_then_pause():
        async def one_req(i: int):
            engine_input = {
                "prompts": [messages],  # single request path
                "prompt_token_ids": None,
                "sampling_params": dict(sampling_params),
                "session_ids": [i],
            }
            return await client.generate(engine_input)

        tasks = [asyncio.create_task(one_req(i)) for i in range(num_requests)]
        # Let requests start and enqueue
        await asyncio.sleep(1)
        # Pause then resume while requests are in-flight
        await client.pause_generation()
        await client.resume_generation()
        # Run for another two seconds, then pause and resume again
        await asyncio.sleep(2)
        await client.pause_generation()
        await client.resume_generation()
        return await asyncio.gather(*tasks)

    outputs = asyncio.run(run_requests_then_pause())

    # 4. Validate each output: stop_reason is "length" and tokens == max_tokens
    assert len(outputs) == num_requests, f"Expected {num_requests} outputs, got {len(outputs)}"
    for i, out in enumerate(outputs):
        # InferenceEngineOutput shape checks
        assert "responses" in out and "response_ids" in out and "stop_reasons" in out
        assert len(out["responses"]) == 1 and len(out["response_ids"]) == 1 and len(out["stop_reasons"]) == 1
        assert out["stop_reasons"][0] == "length", f"Request {i} stop_reason is not 'length': {out['stop_reasons'][0]}"
        # Check completion tokens via response_ids
        token_ids = out["response_ids"][0]
        assert (
            len(token_ids) == sampling_params["max_tokens"]
        ), f"Request {i} expected {sampling_params['max_tokens']} tokens, got {len(token_ids)}"
        # Check response_logprobs length if present
        if "response_logprobs" in out and out["response_logprobs"] is not None:
            assert (
                len(out["response_logprobs"][0]) == sampling_params["max_tokens"]
            ), f"Request {i} expected {sampling_params['max_tokens']} logprobs, got {len(out['response_logprobs'][0])}"
        # Check string output decodes correctly
        assert out["responses"][0] == client.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Print a preview to aid debugging
        print(f"Output first 1500 chars: {out['responses'][0][:1500]}...")


@pytest.mark.sglang
def test_abort_generation_sglang_engine(ray_init_fixture):
    """
    We send 4 requests that are really long via client.generate() and then call pause_generation
    to abort them. We expect the requests to be aborted and return with stop_reason "abort".
    """
    num_requests = 4
    max_tokens = 8192

    # 1. Build engine
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL)
    cfg.trainer.placement.colocate_all = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    # We generate max_tokens tokens and ignore eos.
    sampling_params = {
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    client, _ = init_inference_engines(
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend="sglang",
        model=MODEL,
        num_inference_engines=cfg.generator.num_inference_engines,
        sleep_level=1,
    )

    # 2. Build 4 chat prompts that have no early stops
    convs: List[ConversationType] = [
        [
            {"role": "system", "content": "You are a token generator that keeps talking endlessly."},
            {"role": "user", "content": "Write a very long rambling response without ending."},
        ]
        for _ in range(num_requests)
    ]

    # 3. Fire 4 concurrent requests via client.generate()
    async def run_requests_then_pause():
        async def one_req(i: int):
            engine_input = {
                "prompts": [convs[i]],
                "prompt_token_ids": None,
                "sampling_params": dict(sampling_params),
                "session_ids": [i],
            }
            return await client.generate(engine_input)

        tasks = [asyncio.create_task(one_req(i)) for i in range(num_requests)]
        # Wait to let it run a bit, then pause generation (which aborts)
        await asyncio.sleep(1)
        await client.pause_generation()
        return await asyncio.gather(*tasks)

    outputs = asyncio.run(run_requests_then_pause())

    # 4. Validate outputs: each should have stop_reason "abort"
    assert len(outputs) == num_requests, f"Expected {num_requests} outputs, got {len(outputs)}"
    for i, out in enumerate(outputs):
        assert "responses" in out and "response_ids" in out and "stop_reasons" in out
        assert len(out["stop_reasons"]) == 1
        assert out["stop_reasons"][0] == "abort", f"Request {i} stop_reason is not 'abort': {out['stop_reasons'][0]}"
        # The response_ids should be less than max_tokens since we aborted
        token_ids = out["response_ids"][0]
        assert (
            len(token_ids) < max_tokens
        ), f"Request {i} got {len(token_ids)} tokens, expected less than {max_tokens} due to abort"
        print(f"Request {i} aborted with {len(token_ids)} tokens generated")

    # Unpause for cleanup
    asyncio.run(client.resume_generation())
