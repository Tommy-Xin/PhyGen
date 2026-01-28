#!/usr/bin/env python
"""
Script to add processor and tokenizer files to saved CLIP models.
This is useful for models that were saved before adding processor/tokenizer saving logic.
"""
import os
import argparse
from transformers import AutoImageProcessor, AutoTokenizer

def add_processor_tokenizer(model_path, base_model_path=None):
    """
    Add processor and tokenizer files to a saved CLIP model.
    
    Args:
        model_path: Path to the saved model directory
        base_model_path: Path to the base model (if None, will use default)
    """
    if not os.path.isdir(model_path):
        print(f"Error: {model_path} is not a directory")
        return
    
    # Determine base model path if not provided
    if base_model_path is None:
        # Check if it's 336 or 224 model based on path
        if '336' in model_path:
            base_model_path = '/home/gaiyiming/hjq/xinc/GenHancer/Continuous/pretrained_weights/clip-vit-large-patch14-336'
        else:
            base_model_path = '/home/gaiyiming/hjq/xinc/GenHancer/Continuous/pretrained_weights/clip-vit-large-patch14'
    
    if not os.path.isdir(base_model_path):
        print(f"Error: Base model path {base_model_path} does not exist")
        return
    
    print(f"Loading processor and tokenizer from: {base_model_path}")
    print(f"Adding to model directory: {model_path}")
    
    # Load and save processor
    processor = AutoImageProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(model_path)
    print("✓ Saved processor files")
    
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(model_path)
    
    # Check if tokenizer.json exists in base model and copy it if needed
    tokenizer_json_path = os.path.join(base_model_path, 'tokenizer.json')
    if os.path.exists(tokenizer_json_path):
        import shutil
        shutil.copy2(tokenizer_json_path, model_path)
        print("✓ Saved tokenizer files (including tokenizer.json)")
    else:
        print("✓ Saved tokenizer files")
    
    print(f"\n✓ Successfully added processor and tokenizer to {model_path}")
    
    # List all files in the directory
    print("\nFiles in model directory:")
    for f in sorted(os.listdir(model_path)):
        print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add processor and tokenizer to saved CLIP model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Path to the base model (optional, will auto-detect if not provided)"
    )
    
    args = parser.parse_args()
    add_processor_tokenizer(args.model_path, args.base_model_path)

# python add_processor_tokenizer.py --model_path /home/gaiyiming/hjq/xinc/GenHancer/Continuous/saved_models/clip-vit-large-patch14-336/ --base_model_path /home/gaiyiming/hjq/xinc/GenHancer/Continuous/pretrained_weights/clip-vit-large-patch14-336
# python add_processor_tokenizer.py --model_path /home/gaiyiming/hjq/xinc/GenHancer/Continuous/outputs/output_CLIP_seed99_224_stage2_load100000_all_lr1e-5_gradient_accumulation_4/clip-vit-large-patch14-5000 --base_model_path /home/gaiyiming/hjq/xinc/GenHancer/Continuous/pretrained_weights/clip-vit-large-patch14