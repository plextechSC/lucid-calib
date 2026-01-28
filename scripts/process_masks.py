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


def process_masks(input_dir: str, output_dir: str) -> int:
    """
    Process all masks in the input directory.

    Args:
        input_dir: Directory containing mask subdirectories (one per image)
        output_dir: Directory to save processed masks

    Returns:
        Number of masks processed
    """
    masks_list = os.listdir(input_dir)
    masks_list.sort()  # Sort the list to process directories in order

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print("Error: output_dir already exists!")
        return 0

    total_processed = 0

    for mask_name in masks_list:
        input_dir_sub = os.path.join(input_dir, mask_name)

        # Skip if not a directory (e.g., .DS_Store files on macOS)
        if not os.path.isdir(input_dir_sub):
            continue

        output_dir_sub = os.path.join(output_dir, mask_name)
        os.makedirs(output_dir_sub)

        mask_files = os.listdir(input_dir_sub)
        n = 0

        for file in mask_files:
            if not file.endswith('.png'):
                continue

            mask_ori = cv2.imread(os.path.join(input_dir_sub, file))
            if mask_ori is None:
                continue

            H, W, _ = mask_ori.shape
            mask_ori = mask_ori > 128

            mask_ori = np.asarray(mask_ori[:, :, 0], dtype=np.double)
            n_white = np.sum(mask_ori)
            gx, gy = np.gradient(mask_ori)
            temp_edge = gy * gy + gx * gx

            temp_edge[temp_edge != 0.0] = 1

            if n_white < 0.02 * H * W:
                # Small mask - copy as-is
                copyfile(
                    os.path.join(input_dir_sub, file),
                    os.path.join(output_dir_sub, file)
                )
            else:
                # Large mask - expand edges and intersect
                mask_new1 = np.zeros(mask_ori.shape, dtype=bool)
                margin_inside = int(30 + H * W / n_white)

                for i in range(H):
                    for j in range(W):
                        if temp_edge[i][j] != 0:
                            left = max(j - margin_inside, 0)
                            right = min(j + margin_inside, W - 1)
                            top = max(i - margin_inside, 0)
                            bottom = min(i + margin_inside, H - 1)
                            mask_new1[top:bottom, left:right] = 1

                mask_out = np.zeros(mask_ori.shape)
                mask_out[np.logical_and(mask_ori, mask_new1)] = 255
                mask_out = np.asarray(mask_out, dtype=np.uint8)
                cv2.imwrite(os.path.join(output_dir_sub, file), mask_out)
                n += 1

        if n > 0:
            print(f"Processed {n} masks in {mask_name}")
        total_processed += n

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
