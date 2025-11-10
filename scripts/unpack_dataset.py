#!/usr/bin/env python3
"""
Unpacks the .npy dataset file into individual image files.
Saves images to data/raw/ directory organized by steering direction.
"""

import numpy as np
import cv2
import os
from pathlib import Path
from collections import Counter


def unpack_dataset(npy_file, output_dir="data/raw"):
    """
    Unpack .npy dataset into individual image files.

    Args:
        npy_file: Path to the .npy file containing training data
        output_dir: Directory to save extracted images
    """
    # Load the dataset
    print(f"Loading dataset from {npy_file}...")
    data = np.load(npy_file, allow_pickle=True)
    print(f"Dataset loaded: {len(data)} samples")

    # Print class distribution
    labels = [sample[1] for sample in data]
    label_counts = Counter(labels)
    print(f"\nClass distribution:")
    print(f"  Left (-1):    {label_counts.get(-1, 0)} samples")
    print(f"  Forward (0):  {label_counts.get(0, 0)} samples")
    print(f"  Right (1):    {label_counts.get(1, 0)} samples")

    # Create output directories
    output_path = Path(output_dir)
    left_dir = output_path / "left"
    forward_dir = output_path / "forward"
    right_dir = output_path / "right"

    for dir_path in [left_dir, forward_dir, right_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Counters for each direction
    counters = {-1: 0, 0: 0, 1: 0}
    dir_map = {-1: left_dir, 0: forward_dir, 1: right_dir}

    # Extract and save images
    print(f"\nExtracting images to {output_dir}/...")
    for idx, sample in enumerate(data):
        image = sample[0]  # 64x64 grayscale image
        direction = sample[1]  # -1, 0, or 1

        # Get appropriate directory and counter
        save_dir = dir_map[direction]
        counter = counters[direction]

        # Save image
        filename = f"img_{counter:05d}.png"
        filepath = save_dir / filename
        cv2.imwrite(str(filepath), image)

        counters[direction] += 1

        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} samples...")

    print(f"\nExtraction complete!")
    print(f"  Left turns:   {counters[-1]} images saved to {left_dir}")
    print(f"  Forward:      {counters[0]} images saved to {forward_dir}")
    print(f"  Right turns:  {counters[1]} images saved to {right_dir}")
    print(f"  Total:        {sum(counters.values())} images")


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = "data/training_data-SIZE10000-TIME80557.npy"

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        exit(1)

    unpack_dataset(dataset_path)
