#!/usr/bin/env python3
"""
Check if a model has specific parameters (e.g., proj_z)
Usage: python check_model_parameters.py <model_path>
"""

import sys
import torch
from transformers import AutoModelForCausalLM

def check_model_parameters(model_path, search_term="proj_z"):
    """
    Load a model and check if it contains parameters matching the search term

    Args:
        model_path: Path to the model checkpoint
        search_term: Parameter name to search for (default: "proj_z")
    """
    print("="*60)
    print(f"Loading model from: {model_path}")
    print("="*60)

    try:
        # Load the model (CPU only, no GPU)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load to CPU to avoid GPU memory issues
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        print(f"✓ Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print("="*60)

        # Get all parameter names
        all_params = list(model.named_parameters())
        print(f"Total parameters: {len(all_params)}")
        print("="*60)

        # Search for the specific parameter
        print(f"\nSearching for parameters containing '{search_term}'...")
        print("-"*60)

        found_params = []
        for name, param in all_params:
            if search_term.lower() in name.lower():
                found_params.append((name, param))

        if found_params:
            print(f"✓ Found {len(found_params)} parameter(s) matching '{search_term}':")
            print("")
            for name, param in found_params:
                print(f"  - Name: {name}")
                print(f"    Shape: {param.shape}")
                print(f"    Dtype: {param.dtype}")
                print(f"    Device: {param.device}")
                print("")
        else:
            print(f"✗ No parameters found containing '{search_term}'")
            print("")
            print("Here are all parameter names (first 20):")
            print("-"*60)
            for i, (name, param) in enumerate(all_params[:20]):
                print(f"  {i+1}. {name} - {param.shape}")
            if len(all_params) > 20:
                print(f"  ... and {len(all_params) - 20} more")

        print("="*60)
        print("Check complete!")
        print("="*60)

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Default model path - your trained checkpoint
    default_model_path = "/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20"

    # Or use base model:
    # default_model_path = "/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20"

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using provided model path: {model_path}")
    else:
        print(f"No model path provided, using default checkpoint")
        print(f"Default: {default_model_path}")
        print("")
        print("Usage: python check_model_parameters.py <model_path> [search_term]")
        print("Example: python check_model_parameters.py /path/to/model proj_z")
        print("")
        model_path = default_model_path

    # Optional: search for different parameter name
    search_term = sys.argv[2] if len(sys.argv) > 2 else "proj_z"

    print(f"Searching for parameters containing: '{search_term}'")
    print("")

    check_model_parameters(model_path, search_term)
