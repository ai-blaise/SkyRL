#!/usr/bin/env python3
"""
Real MoE inference tests on B200 8-GPU cluster.

Run with:
    pytest tests/gpu/test_real_moe_inference.py -v -s
    
Or standalone:
    python tests/gpu/test_real_moe_inference.py
"""
import os
import time
import pytest
import torch


def get_gpu_info():
    """Get GPU configuration."""
    num_gpus = torch.cuda.device_count()
    total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus))
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    return {
        "num_gpus": num_gpus,
        "total_vram_gb": total_vram / 1e9,
        "gpu_names": gpu_names,
    }


def run_inference_benchmark(model_path: str, tp_size: int, prompts: list[str], max_tokens: int = 1024):
    """Run inference benchmark and return results."""
    from sglang import Engine
    
    print(f"\n{'='*80}")
    print(f"Model: {model_path}")
    print(f"Tensor Parallel Size: {tp_size}")
    print(f"{'='*80}\n")
    
    start = time.time()
    engine = Engine(
        model_path=model_path,
        tp_size=tp_size,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        trust_remote_code=True,
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s\n")
    
    results = []
    total_tokens = 0
    total_time = 0
    
    for i, prompt in enumerate(prompts):
        print(f"[Prompt {i+1}]: {prompt[:100]}...")
        
        start = time.time()
        response = engine.generate(
            prompt,
            sampling_params={
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
        elapsed = time.time() - start
        
        output = response['text']
        num_tokens = len(output.split()) * 1.3  # Rough estimate
        tokens_per_sec = num_tokens / elapsed
        
        total_tokens += num_tokens
        total_time += elapsed
        
        print(f"[Response]: {num_tokens:.0f} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        print(f"{'-'*80}")
        print(output[:500])
        if len(output) > 500:
            print(f"... [{len(output)-500} more chars]")
        print(f"{'-'*80}\n")
        
        results.append({
            "prompt": prompt,
            "output": output,
            "tokens": num_tokens,
            "time": elapsed,
            "tokens_per_sec": tokens_per_sec,
        })
    
    engine.shutdown()
    
    return {
        "model": model_path,
        "tp_size": tp_size,
        "load_time": load_time,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "avg_throughput": total_tokens / total_time,
        "results": results,
    }


# Test prompts for MoE models
MOE_PROMPTS = [
    "Explain the key differences between Mixture-of-Experts (MoE) models and dense transformer models. Cover the gating mechanism, computational efficiency, and scaling properties.",
    
    "Write a Python function that implements a simple top-k gating mechanism for a Mixture-of-Experts layer. Include proper softmax normalization and load balancing.",
    
    "What are the main trade-offs between Expert Parallelism (EP) and Tensor Parallelism (TP) when serving large MoE models? Consider memory, communication, and latency.",
]

TECHNICAL_PROMPTS = [
    """You are an expert AI researcher. Provide a detailed mathematical formulation of the Mixture-of-Experts layer, including:
1. The gating function G(x) with softmax and top-k selection
2. The load balancing auxiliary loss
3. How expert parallelism distributes computation across devices""",

    """Implement a CUDA kernel for fused attention with the flash attention algorithm. Include:
- Memory coalescing
- Shared memory tiling  
- Online softmax for numerical stability
Add comments explaining each optimization.""",
]


@pytest.mark.gpu
@pytest.mark.slow
class TestRealMoEInference:
    """Real MoE inference tests on B200 cluster."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.gpu_info = get_gpu_info()
        print(f"\nGPU Config: {self.gpu_info['num_gpus']}x {self.gpu_info['gpu_names'][0]}")
        print(f"Total VRAM: {self.gpu_info['total_vram_gb']:.0f} GB")
    
    @pytest.mark.skipif(
        torch.cuda.device_count() < 1,
        reason="No GPU available"
    )
    def test_qwen_7b_single_gpu(self):
        """Test Qwen 7B on single GPU."""
        results = run_inference_benchmark(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            tp_size=1,
            prompts=MOE_PROMPTS[:2],
            max_tokens=512,
        )
        assert results["avg_throughput"] > 50, "Throughput too low"
        assert all(len(r["output"]) > 100 for r in results["results"]), "Output too short"
    
    @pytest.mark.skipif(
        torch.cuda.device_count() < 8,
        reason="Need 8 GPUs for this test"
    )
    def test_deepseek_v2_lite_8gpu(self):
        """Test DeepSeek-V2-Lite MoE on 8 GPUs."""
        results = run_inference_benchmark(
            model_path="deepseek-ai/DeepSeek-V2-Lite",
            tp_size=8,
            prompts=MOE_PROMPTS,
            max_tokens=1024,
        )
        assert results["avg_throughput"] > 100, "Throughput too low for 8 GPU"
        assert all(len(r["output"]) > 200 for r in results["results"]), "Output too short"
    
    @pytest.mark.skipif(
        torch.cuda.device_count() < 8 or get_gpu_info()["total_vram_gb"] < 1000,
        reason="Need 8 GPUs with >1TB VRAM for DeepSeek-V3"
    )
    def test_deepseek_v3_8gpu(self):
        """Test DeepSeek-V3 (685B) on 8x B200."""
        results = run_inference_benchmark(
            model_path="deepseek-ai/DeepSeek-V3",
            tp_size=8,
            prompts=TECHNICAL_PROMPTS,
            max_tokens=2048,
        )
        assert results["avg_throughput"] > 20, "Throughput too low for 685B model"
        assert all(len(r["output"]) > 500 for r in results["results"]), "Output too short"


def main():
    """Run standalone benchmark."""
    gpu_info = get_gpu_info()
    print("=" * 80)
    print("B200 MoE INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"GPUs: {gpu_info['num_gpus']}x {gpu_info['gpu_names'][0]}")
    print(f"Total VRAM: {gpu_info['total_vram_gb']:.0f} GB")
    print("=" * 80)
    
    # Choose model based on available resources
    if gpu_info["num_gpus"] >= 8 and gpu_info["total_vram_gb"] > 1000:
        # Full power - run DeepSeek-V3
        print("\nRunning DeepSeek-V3 (685B) on 8x B200...")
        results = run_inference_benchmark(
            model_path="deepseek-ai/DeepSeek-V3",
            tp_size=8,
            prompts=TECHNICAL_PROMPTS,
            max_tokens=2048,
        )
    elif gpu_info["num_gpus"] >= 8:
        # 8 GPUs but less VRAM - run DeepSeek-V2-Lite
        print("\nRunning DeepSeek-V2-Lite on 8 GPUs...")
        results = run_inference_benchmark(
            model_path="deepseek-ai/DeepSeek-V2-Lite", 
            tp_size=8,
            prompts=MOE_PROMPTS,
            max_tokens=1024,
        )
    else:
        # Fallback to Qwen
        print(f"\nRunning Qwen 7B on {gpu_info['num_gpus']} GPU(s)...")
        results = run_inference_benchmark(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            tp_size=min(gpu_info["num_gpus"], 2),
            prompts=MOE_PROMPTS[:2],
            max_tokens=512,
        )
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Model: {results['model']}")
    print(f"TP Size: {results['tp_size']}")
    print(f"Load Time: {results['load_time']:.1f}s")
    print(f"Total Tokens: {results['total_tokens']:.0f}")
    print(f"Total Time: {results['total_time']:.1f}s")
    print(f"Avg Throughput: {results['avg_throughput']:.1f} tokens/sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
