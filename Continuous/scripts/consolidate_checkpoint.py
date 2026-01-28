#!/usr/bin/env python3
"""
Consolidate multiple checkpoint files from GenHancer stage1 training into a single 
weight file that can be directly used for inference.

Usage:
    python consolidate_checkpoint.py \
        --checkpoint_dir  \
        --step 100000 \
        --output_path /path/to/consolidated_checkpoint.bin
"""

import argparse
import os
import torch
from collections import OrderedDict


def consolidate_checkpoint(checkpoint_dir, step, output_path):
    """
    Consolidate multiple checkpoint files into a single weight file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        step: Checkpoint step number to consolidate (e.g., 100000)
        output_path: Path to output consolidated checkpoint file
    """
    print(f"Consolidating checkpoint at step: {step}")
    print(f"Input directory: {checkpoint_dir}")
    print(f"Output path: {output_path}")
    
    # Build checkpoint file paths
    dit_path = os.path.join(checkpoint_dir, f"checkpoint-dit-{step}.bin")
    project_clip_path = os.path.join(checkpoint_dir, f"checkpoint-project-clip-{step}.bin")
    project_t5_path = os.path.join(checkpoint_dir, f"checkpoint-project-t5-{step}.bin")
    
    # Check if files exist
    missing_files = []
    for name, path in [("dit", dit_path), ("project_clip", project_clip_path), ("project_t5", project_t5_path)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        raise FileNotFoundError(f"The following checkpoint files do not exist:\n" + "\n".join(missing_files))
    
    # Load each checkpoint
    print(f"\nLoading checkpoint files...")
    dit_state_dict = torch.load(dit_path, map_location='cpu')
    print(f"  ✓ Loaded dit checkpoint: {len(dit_state_dict)} parameters")
    
    project_clip_state_dict = torch.load(project_clip_path, map_location='cpu')
    print(f"  ✓ Loaded project_clip checkpoint: {len(project_clip_state_dict)} parameters")
    
    project_t5_state_dict = torch.load(project_t5_path, map_location='cpu')
    print(f"  ✓ Loaded project_t5 checkpoint: {len(project_t5_state_dict)} parameters")
    
    # Consolidate weights into a single dictionary
    consolidated_dict = OrderedDict()
    
    # Add dit weights (keep original keys, as dit is an independent model)
    print(f"\nConsolidating weights...")
    for key, value in dit_state_dict.items():
        consolidated_dict[f"dit.{key}"] = value
    print(f"  ✓ Added dit weights: {len(dit_state_dict)} parameters")
    
    # Add project_clip weights (add clip_vis.project_clip prefix)
    for key, value in project_clip_state_dict.items():
        consolidated_dict[f"clip_vis.project_clip.{key}"] = value
    print(f"  ✓ Added project_clip weights: {len(project_clip_state_dict)} parameters")
    
    # Add project_t5 weights (add clip_vis.project_t5 prefix)
    for key, value in project_t5_state_dict.items():
        consolidated_dict[f"clip_vis.project_t5.{key}"] = value
    print(f"  ✓ Added project_t5 weights: {len(project_t5_state_dict)} parameters")
    
    # Save consolidated weights (ensure original files are not overwritten)
    print(f"\nSaving consolidated weights to: {output_path}")
    if os.path.exists(output_path):
        # If file exists, add timestamp suffix
        import time
        timestamp = int(time.time())
        base_name, ext = os.path.splitext(output_path)
        output_path = f"{base_name}_{timestamp}{ext}"
        print(f"  Note: File already exists, using new filename: {output_path}")
    
    torch.save(consolidated_dict, output_path)
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Save completed! File size: {file_size_mb:.2f} MB")
    print(f"  ✓ Total parameters: {len(consolidated_dict)}")
    
    # Verify saved file
    print(f"\nVerifying saved file...")
    loaded_dict = torch.load(output_path, map_location='cpu')
    if len(loaded_dict) == len(consolidated_dict):
        print(f"  ✓ Verification successful! File contains {len(loaded_dict)} parameters")
    else:
        print(f"  ✗ Verification failed! Expected {len(consolidated_dict)} parameters, got {len(loaded_dict)}")
    
    print(f"\nConsolidation completed!")
    print(f"You can load the weights using the following code:")
    print(f"  checkpoint = torch.load('{output_path}', map_location='cpu')")
    print(f"  dit.load_state_dict({{k.replace('dit.', ''): v for k, v in checkpoint.items() if k.startswith('dit.')}})")
    print(f"  clip_vis.project_clip.load_state_dict({{k.replace('clip_vis.project_clip.', ''): v for k, v in checkpoint.items() if k.startswith('clip_vis.project_clip.')}})")
    print(f"  clip_vis.project_t5.load_state_dict({{k.replace('clip_vis.project_t5.', ''): v for k, v in checkpoint.items() if k.startswith('clip_vis.project_t5.')}})")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate multiple checkpoint files from GenHancer stage1 training into a single weight file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint files"
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Checkpoint step number to consolidate (e.g., 100000)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to output consolidated checkpoint file (default: checkpoint_dir/consolidated_checkpoint_step{step}.bin, will not overwrite original files)"
    )
    
    args = parser.parse_args()
    
    # If output path not specified, use default path (ensure original files are not overwritten)
    if args.output_path is None:
        args.output_path = os.path.join(
            args.checkpoint_dir,
            f"consolidated_checkpoint_step{args.step}.bin"
        )
    
    # Check if output path conflicts with original files
    original_files = [
        os.path.join(args.checkpoint_dir, f"checkpoint-dit-{args.step}.bin"),
        os.path.join(args.checkpoint_dir, f"checkpoint-project-clip-{args.step}.bin"),
        os.path.join(args.checkpoint_dir, f"checkpoint-project-t5-{args.step}.bin"),
    ]
    
    if os.path.abspath(args.output_path) in [os.path.abspath(f) for f in original_files]:
        raise ValueError(
            f"Error: Output path '{args.output_path}' conflicts with original checkpoint files!\n"
            f"Please use a different filename, e.g., consolidated_checkpoint_step{args.step}.bin"
        )
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    consolidate_checkpoint(args.checkpoint_dir, args.step, args.output_path)


if __name__ == "__main__":
    main()

