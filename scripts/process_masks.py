#!/usr/bin/env python3
"""
Process SAM-generated masks to filter and refine them.

This script:
1. Loads SAM masks (binary PNG format)
2. Computes gradients to find mask edges
3. Filters masks based on size:
   - If mask is <2% of image area: keep as-is (small object)
   - If mask is >2%: expand edge region and keep only intersection
4. Outputs refined masks maintaining original dimensions

Usage:
    python scripts/process_masks.py -i masks_dir -o processed_masks_dir

Example:
    python scripts/process_masks.py -i data/nuscenes/masks -o data/nuscenes/processed_masks
"""

import numpy as np
import cv2
import os
from shutil import copyfile
import argparse


def process_masks(input_dir: str, output_dir: str, verbose: bool = True) -> int:
    """
    Process all masks in the input directory.

    Args:
        input_dir: Directory containing mask subdirectories (one per image)
        output_dir: Directory to save processed masks
        verbose: Print progress messages

    Returns:
        Number of masks processed
    """
    masks_list = os.listdir(input_dir)
    masks_list.sort()  # Sort the list to process directories in order
    
    # Filter to only directories
    masks_list = [m for m in masks_list if os.path.isdir(os.path.join(input_dir, m))]

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print("Error: output_dir already exists!")
        return 0

    total_processed = 0
    total_dirs = len(masks_list)

    for dir_idx, mask_name in enumerate(masks_list):
        input_dir_sub = os.path.join(input_dir, mask_name)
        output_dir_sub = os.path.join(output_dir, mask_name)
        os.makedirs(output_dir_sub)

        mask_files = [f for f in os.listdir(input_dir_sub) if f.endswith('.png')]
        mask_files.sort()
        n_processed = 0
        n_copied = 0
        
        if verbose:
            print(f"[{dir_idx+1}/{total_dirs}] Processing {mask_name} ({len(mask_files)} masks)...")

        for file_idx, file in enumerate(mask_files):
            mask_ori = cv2.imread(os.path.join(input_dir_sub, file))
            if mask_ori is None:
                if verbose:
                    print(f"  Warning: Could not read {file}")
                continue

            H, W, _ = mask_ori.shape
            mask_ori = mask_ori > 128
            mask_ori = np.asarray(mask_ori[:, :, 0], dtype=np.uint8)
            n_white = np.sum(mask_ori)

            if n_white < 0.02 * H * W:
                # Small mask - copy as-is
                copyfile(
                    os.path.join(input_dir_sub, file),
                    os.path.join(output_dir_sub, file)
                )
                n_copied += 1
            else:
                # Large mask - expand edges using morphological dilation (FAST)
                # Calculate margin based on mask density
                margin_inside = int(30 + H * W / n_white)
                
                # Find edges using gradient
                gx, gy = np.gradient(mask_ori.astype(np.float32))
                temp_edge = (gy * gy + gx * gx) > 0
                temp_edge = temp_edge.astype(np.uint8)
                
                # Use morphological dilation instead of nested loops (MUCH faster)
                kernel_size = margin_inside * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                mask_dilated = cv2.dilate(temp_edge, kernel, iterations=1)
                
                # Intersect with original mask
                mask_out = np.zeros(mask_ori.shape, dtype=np.uint8)
                mask_out[np.logical_and(mask_ori, mask_dilated)] = 255
                cv2.imwrite(os.path.join(output_dir_sub, file), mask_out)
                n_processed += 1

        if verbose:
            print(f"  -> {n_processed} processed, {n_copied} copied (small)")
        total_processed += n_processed + n_copied

    return total_processed


def main():
    parser = argparse.ArgumentParser(
        prog='process_masks',
        description='Process and filter SAM-generated masks'
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input masks folder path')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output processed masks folder path')

    args = parser.parse_args()

    total = process_masks(args.input, args.output)
    print(f"\nTotal masks processed: {total}")

    return 0


if __name__ == "__main__":
    exit(main())
