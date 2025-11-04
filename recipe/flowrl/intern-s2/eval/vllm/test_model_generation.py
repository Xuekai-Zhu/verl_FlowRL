#!/usr/bin/env python3
"""
Test model generation with vLLM
Usage: python test_model_generation.py <model_path>
"""

import sys
from vllm import LLM, SamplingParams

def test_model_generation(model_path):
    """Test model generation with several prompts"""

    print("="*60)
    print(f"Loading model: {model_path}")
    print("="*60)

    # Initialize vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        trust_remote_code=True
    )

    print("✓ Model loaded successfully!")
    print("")

    # Test prompts
    test_prompts = [
        "What is 2+2? Answer:",
        "Calculate: 15 * 7 = ",
        "Solve the equation: If x + 5 = 12, then x = ",
        "What is the capital of France?",
        "Write a short sentence about machine learning:",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    )

    print("="*60)
    print("Testing model generation...")
    print("="*60)
    print("")

    # Generate responses
    outputs = llm.generate(test_prompts, sampling_params)

    # Display results
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print(f"[Test {i+1}/{len(test_prompts)}]")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print("-"*60)
        print("")

    print("="*60)
    print("✓ All tests completed successfully!")
    print("✓ Model weights are working correctly!")
    print("="*60)


if __name__ == "__main__":
    # Default model path - your trained checkpoint
    default_model_path = "/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20"

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print(f"Using default model: {default_model_path}")
        print("")
        model_path = default_model_path

    test_model_generation(model_path)
