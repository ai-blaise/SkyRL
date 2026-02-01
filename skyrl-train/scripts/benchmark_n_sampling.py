#!/usr/bin/env python3
"""Benchmark n>1 sampling vs N separate requests.

This script compares the performance of:
1. Single request with n=k samples (parallel sampling, shared prefill)
2. k separate requests with n=1 each (sequential, repeated prefill)

Usage:
    # Basic benchmark with SGLang
    python scripts/benchmark_n_sampling.py --model Qwen/Qwen2.5-0.5B-Instruct --n 4

    # With custom parameters
    python scripts/benchmark_n_sampling.py --model meta-llama/Llama-3.2-1B-Instruct \
        --n 8 --num-prompts 10 --max-tokens 128 --warmup 2

    # Compare multiple n values
    python scripts/benchmark_n_sampling.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --n 2 4 8 16 --num-prompts 5
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str  # "n_sampling" or "separate_requests"
    n_samples: int
    num_prompts: int
    total_outputs: int
    total_time_ms: float
    avg_time_per_prompt_ms: float
    avg_time_per_output_ms: float
    throughput_outputs_per_sec: float
    total_tokens_generated: int
    tokens_per_second: float


@dataclass
class ComparisonResult:
    """Comparison between n>1 sampling and separate requests."""
    n_samples: int
    n_sampling: BenchmarkResult
    separate: BenchmarkResult
    speedup: float  # n_sampling speedup over separate
    time_saved_ms: float
    time_saved_pct: float


def create_test_prompts(num_prompts: int, tokenizer) -> List[List[int]]:
    """Create test prompts as token IDs."""
    test_messages = [
        [{"role": "user", "content": "Write a short poem about the ocean."}],
        [{"role": "user", "content": "Explain quantum computing in simple terms."}],
        [{"role": "user", "content": "What are the benefits of exercise?"}],
        [{"role": "user", "content": "Describe a beautiful sunset."}],
        [{"role": "user", "content": "How does photosynthesis work?"}],
        [{"role": "user", "content": "Write a haiku about mountains."}],
        [{"role": "user", "content": "What is machine learning?"}],
        [{"role": "user", "content": "Explain the water cycle."}],
    ]

    # Cycle through test messages if we need more
    prompts = []
    for i in range(num_prompts):
        messages = test_messages[i % len(test_messages)]
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompts.append(token_ids)

    return prompts


async def benchmark_n_sampling(
    engine,
    prompts: List[List[int]],
    n_samples: int,
    max_tokens: int,
    temperature: float,
) -> BenchmarkResult:
    """Benchmark single request with n>1 samples."""
    from skyrl_train.inference_engines.base import InferenceEngineInput

    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "n": n_samples,
    }

    input_batch = InferenceEngineInput(
        prompts=None,
        prompt_token_ids=prompts,
        sampling_params=sampling_params,
        session_ids=None,
    )

    start = time.perf_counter()
    output = await engine.generate(input_batch)
    elapsed_ms = (time.perf_counter() - start) * 1000

    total_outputs = len(output["responses"])
    total_tokens = sum(len(ids) for ids in output["response_ids"])

    return BenchmarkResult(
        method="n_sampling",
        n_samples=n_samples,
        num_prompts=len(prompts),
        total_outputs=total_outputs,
        total_time_ms=elapsed_ms,
        avg_time_per_prompt_ms=elapsed_ms / len(prompts),
        avg_time_per_output_ms=elapsed_ms / total_outputs,
        throughput_outputs_per_sec=total_outputs / (elapsed_ms / 1000),
        total_tokens_generated=total_tokens,
        tokens_per_second=total_tokens / (elapsed_ms / 1000),
    )


async def benchmark_separate_requests(
    engine,
    prompts: List[List[int]],
    n_samples: int,
    max_tokens: int,
    temperature: float,
) -> BenchmarkResult:
    """Benchmark N separate requests with n=1 each."""
    from skyrl_train.inference_engines.base import InferenceEngineInput

    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
    }

    total_outputs = 0
    total_tokens = 0

    start = time.perf_counter()

    # Run n_samples separate batches
    for _ in range(n_samples):
        input_batch = InferenceEngineInput(
            prompts=None,
            prompt_token_ids=prompts,
            sampling_params=sampling_params,
            session_ids=None,
        )
        output = await engine.generate(input_batch)
        total_outputs += len(output["responses"])
        total_tokens += sum(len(ids) for ids in output["response_ids"])

    elapsed_ms = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        method="separate_requests",
        n_samples=n_samples,
        num_prompts=len(prompts),
        total_outputs=total_outputs,
        total_time_ms=elapsed_ms,
        avg_time_per_prompt_ms=elapsed_ms / len(prompts),
        avg_time_per_output_ms=elapsed_ms / total_outputs,
        throughput_outputs_per_sec=total_outputs / (elapsed_ms / 1000),
        total_tokens_generated=total_tokens,
        tokens_per_second=total_tokens / (elapsed_ms / 1000),
    )


def compare_results(n_result: BenchmarkResult, sep_result: BenchmarkResult) -> ComparisonResult:
    """Compare n>1 sampling vs separate requests."""
    speedup = sep_result.total_time_ms / n_result.total_time_ms
    time_saved = sep_result.total_time_ms - n_result.total_time_ms
    time_saved_pct = (time_saved / sep_result.total_time_ms) * 100

    return ComparisonResult(
        n_samples=n_result.n_samples,
        n_sampling=n_result,
        separate=sep_result,
        speedup=speedup,
        time_saved_ms=time_saved,
        time_saved_pct=time_saved_pct,
    )


def print_result(result: BenchmarkResult):
    """Print a single benchmark result."""
    print(f"  Method: {result.method}")
    print(f"  Prompts: {result.num_prompts}, Samples/prompt: {result.n_samples}")
    print(f"  Total outputs: {result.total_outputs}")
    print(f"  Total time: {result.total_time_ms:.1f}ms")
    print(f"  Avg time/prompt: {result.avg_time_per_prompt_ms:.1f}ms")
    print(f"  Avg time/output: {result.avg_time_per_output_ms:.1f}ms")
    print(f"  Throughput: {result.throughput_outputs_per_sec:.1f} outputs/sec")
    print(f"  Tokens generated: {result.total_tokens_generated}")
    print(f"  Token throughput: {result.tokens_per_second:.1f} tokens/sec")


def print_comparison(comp: ComparisonResult):
    """Print comparison results."""
    print(f"\n{'='*60}")
    print(f"COMPARISON: n={comp.n_samples} sampling vs {comp.n_samples} separate requests")
    print(f"{'='*60}")

    print(f"\nn={comp.n_samples} Parallel Sampling (shared prefill):")
    print_result(comp.n_sampling)

    print(f"\n{comp.n_samples} Separate Requests (repeated prefill):")
    print_result(comp.separate)

    print(f"\n--- SUMMARY ---")
    print(f"Speedup: {comp.speedup:.2f}x")
    print(f"Time saved: {comp.time_saved_ms:.1f}ms ({comp.time_saved_pct:.1f}%)")

    if comp.speedup > 1:
        print(f"Result: n>1 sampling is {comp.speedup:.2f}x FASTER")
    else:
        print(f"Result: Separate requests are {1/comp.speedup:.2f}x faster (unexpected)")


async def run_benchmark(
    model: str,
    n_values: List[int],
    num_prompts: int,
    max_tokens: int,
    temperature: float,
    warmup_runs: int,
    tp_size: int,
):
    """Run the full benchmark suite."""
    from transformers import AutoTokenizer
    from sglang import Engine
    from skyrl_train.inference_engines.sglang.sglang_engine import SGLangInferenceEngine

    print(f"Loading model: {model}")
    print(f"Tensor parallel size: {tp_size}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    # Initialize SGLang engine
    engine = Engine(
        model_path=model,
        tp_size=tp_size,
        trust_remote_code=True,
    )

    # Wrap in SkyRL interface
    sglang_engine = SGLangInferenceEngine(engine=engine, tokenizer=tokenizer)

    # Create test prompts
    prompts = create_test_prompts(num_prompts, tokenizer)
    print(f"Created {len(prompts)} test prompts")

    # Warmup runs
    if warmup_runs > 0:
        print(f"\nRunning {warmup_runs} warmup iterations...")
        for i in range(warmup_runs):
            await benchmark_n_sampling(sglang_engine, prompts[:2], 2, max_tokens, temperature)
        print("Warmup complete.")

    # Run benchmarks for each n value
    results: List[ComparisonResult] = []

    for n in n_values:
        print(f"\n{'#'*60}")
        print(f"Benchmarking n={n}")
        print(f"{'#'*60}")

        # Benchmark n>1 sampling
        print(f"\nRunning n={n} parallel sampling...")
        n_result = await benchmark_n_sampling(
            sglang_engine, prompts, n, max_tokens, temperature
        )

        # Benchmark separate requests
        print(f"Running {n} separate requests...")
        sep_result = await benchmark_separate_requests(
            sglang_engine, prompts, n, max_tokens, temperature
        )

        # Compare
        comparison = compare_results(n_result, sep_result)
        results.append(comparison)
        print_comparison(comparison)

    # Print final summary table
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Prompts: {num_prompts}, Max tokens: {max_tokens}, Temperature: {temperature}")
    print()
    print(f"{'n':>4} | {'n>1 (ms)':>10} | {'Sep (ms)':>10} | {'Speedup':>8} | {'Saved':>8}")
    print(f"{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for r in results:
        print(f"{r.n_samples:>4} | {r.n_sampling.total_time_ms:>10.1f} | {r.separate.total_time_ms:>10.1f} | {r.speedup:>7.2f}x | {r.time_saved_pct:>6.1f}%")

    # Cleanup
    engine.shutdown()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark n>1 sampling vs N separate requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        nargs="+",
        default=[4],
        help="Number of samples per prompt to benchmark (can specify multiple)",
    )
    parser.add_argument(
        "--num-prompts", "-p",
        type=int,
        default=5,
        help="Number of prompts to use in each benchmark",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=64,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (must be >0 for n>1)",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=2,
        help="Number of warmup runs before benchmarking",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        print("Warning: temperature=0 with n>1 produces identical outputs. Using temperature=0.8")
        args.temperature = 0.8

    asyncio.run(run_benchmark(
        model=args.model,
        n_values=args.n,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        warmup_runs=args.warmup,
        tp_size=args.tp_size,
    ))


if __name__ == "__main__":
    main()
