#!/usr/bin/env python3
"""
Remove proj_z parameters from a saved HuggingFace model checkpoint
Usage: python remove_proj_z.py <input_model_path> [output_model_path]
"""

import sys
import os
import shutil
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights


def remove_proj_z_params(input_path, output_path=None):
    """
    Remove proj_z.* parameters from model checkpoint

    Args:
        input_path: Path to the input model checkpoint
        output_path: Path to save cleaned model (if None, saves to input_path_no_proj_z)
    """
    input_path = Path(input_path)

    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.name}_no_proj_z"
    else:
        output_path = Path(output_path)

    print("="*60)
    print(f"Input model: {input_path}")
    print(f"Output model: {output_path}")
    print("="*60)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Only use safetensors format
    print("\nUsing safetensors format (default)")
    from safetensors.torch import load_file

    safetensor_files = list(input_path.glob("*.safetensors"))

    if not safetensor_files:
        print("✗ No safetensors files found!")
        print("This script only works with safetensors format.")
        return

    # Copy config files and tokenizer (exclude .bin files)
    print("\nCopying config and tokenizer files...")
    for file in input_path.glob("*"):
        if file.suffix in ['.json', '.txt', '.model', '.py'] or file.name.endswith('.jinja'):
            print(f"  Copying: {file.name}")
            shutil.copy2(file, output_path / file.name)

    # Load and clean model state dict
    print("\nProcessing model weights...")

    # Load state dict from safetensors
    state_dict = {}
    for model_file in safetensor_files:
        print(f"  Loading: {model_file.name}")
        loaded = load_file(str(model_file))
        state_dict.update(loaded)

    print(f"\nTotal parameters before: {len(state_dict)}")

    # Remove proj_z parameters
    proj_z_params = [k for k in state_dict.keys() if 'proj_z' in k.lower()]
    print(f"Found {len(proj_z_params)} proj_z parameters:")
    for param in proj_z_params:
        print(f"  - {param}")
        del state_dict[param]

    print(f"\nTotal parameters after: {len(state_dict)}")

    # Save cleaned state dict using HuggingFace save_pretrained
    # This automatically handles splitting large models into multiple files
    print(f"\nSaving cleaned model to: {output_path}")

    # Load model config and create empty model
    config = AutoConfig.from_pretrained(input_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")

    # Use save_pretrained which automatically handles large models
    model.save_pretrained(output_path, state_dict=state_dict)
    del model
    del state_dict

    print("\n" + "="*60)
    print("✓ Successfully removed proj_z parameters!")
    print(f"✓ Cleaned model saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    # Default paths
    default_input = "/mnt/shared-storage-user/llmit/user/xuekaizhu/verl_FlowRL/work_dirs/FlowRL_Scaling/FlowRL-Qwen2.5-0.5B-DAPO-Math-prompt-modified-reward/20251103_090837/huggingface/global_step_20"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("No input path provided, using default:")
        print(f"  {default_input}")
        print("\nUsage: python remove_proj_z.py <input_model_path> [output_model_path]")
        print("")
        input_path = default_input

    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    remove_proj_z_params(input_path, output_path)
