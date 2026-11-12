"""Download MNIST with the Hugging Face `datasets` package and save a subset
of the digits 0, 1, 2, 3, 4 as two PyTorch files:

    data/mnist_train.pt :  100 images per digit ( 500 total), for training
    data/mnist_test.pt  : 1000 images per digit (5000 total), for testing

The 28 x 28 images of MNIST are resized to 32 x 32 (bilinear interpolation).

Each file holds a dict of two tensors:
    "images" : torch.uint8, shape (N, 1, 32, 32), gray levels 0..255
    "labels" : torch.int64, shape (N,), values 0..4

The images of a file are ordered by digit: all the 0s, then all the 1s, ...

Where the images come from:
    training : the first 100 images of each digit in the MNIST train split.
    testing  : the first 1000 images of each digit in the MNIST test split.
               That split has fewer than 1000 images of the digits 0 (980)
               and 4 (982): the missing ones are the images of the MNIST
               train split that follow the 100 training images of the digit.
    No image of MNIST is in both files.

Requirements:
    pip install datasets numpy pillow torch

Usage:
    python util/prepare_data_mnist.py

Load a file back with:
    data = torch.load("data/mnist_train.pt")
    images, labels = data["images"], data["labels"]
    x = images.float().div(255)    # (N, 1, 32, 32), values in [0, 1]
"""

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

CLASSES = [0, 1, 2, 3, 4]

# The images are resized to IMAGE_SIZE x IMAGE_SIZE
IMAGE_SIZE = 32

# Number of images to save for each class
NUM_TRAIN_PER_CLASS = 100
NUM_TEST_PER_CLASS = 1000

# <project root>/data, independent of the directory the script is run from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_images(dataset, indices):
    """Return the images of `dataset` at the positions `indices`, resized, as
    a uint8 tensor of shape (len(indices), 1, IMAGE_SIZE, IMAGE_SIZE)."""
    images = []
    for example in dataset.select(indices):
        image = example["image"].convert("L")    # PIL image, 8-bit gray scale
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        images.append(np.array(image, dtype=np.uint8))
    return torch.from_numpy(np.stack(images)).unsqueeze(1)


def build_subsets():
    """Return {"train": (images, labels), "test": (images, labels)}, the
    images and the labels to save in each file."""
    # Downloads on first use, then reads from the local Hugging Face cache.
    mnist = {split: load_dataset("ylecun/mnist", split=split) for split in ("train", "test")}

    # Reading only the label column avoids decoding every image of MNIST.
    mnist_labels = {split: np.array(dataset["label"]) for split, dataset in mnist.items()}

    images = {"train": [], "test": []}
    for digit in CLASSES:
        # positions of the images of this digit in each split of MNIST
        in_train = np.flatnonzero(mnist_labels["train"] == digit)
        in_test = np.flatnonzero(mnist_labels["test"] == digit)

        # Test images that the MNIST test split cannot provide: they are taken
        # from the MNIST train split, after the images used for training.
        num_extra = max(0, NUM_TEST_PER_CLASS - len(in_test))
        if len(in_train) < NUM_TRAIN_PER_CLASS + num_extra:
            raise ValueError(
                f"MNIST has only {len(in_train)} + {len(in_test)} images of digit {digit}, "
                f"{NUM_TRAIN_PER_CLASS} + {NUM_TEST_PER_CLASS} requested"
            )

        train_images = read_images(mnist["train"], in_train[:NUM_TRAIN_PER_CLASS])
        test_images = read_images(mnist["test"], in_test[:NUM_TEST_PER_CLASS])
        if num_extra > 0:
            extra = in_train[NUM_TRAIN_PER_CLASS:NUM_TRAIN_PER_CLASS + num_extra]
            test_images = torch.cat([test_images, read_images(mnist["train"], extra)])
            print(f"digit {digit}: {len(in_test)} test images from the MNIST test split, "
                  f"{num_extra} from the MNIST train split")

        images["train"].append(train_images)
        images["test"].append(test_images)

    subsets = {}
    for split, images_per_digit in images.items():
        labels = [torch.full((len(x),), digit, dtype=torch.int64)
                  for digit, x in zip(CLASSES, images_per_digit)]
        subsets[split] = (torch.cat(images_per_digit), torch.cat(labels))
    return subsets


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, (images, labels) in build_subsets().items():
        out_file = DATA_DIR / f"mnist_{split}.pt"
        torch.save({"images": images, "labels": labels}, out_file)

        print(f"saved {out_file}")
        print(f"  images: shape={tuple(images.shape)}, dtype={images.dtype}, "
              f"range=[{images.min().item()}, {images.max().item()}]")
        print(f"  labels: shape={tuple(labels.shape)}, dtype={labels.dtype}")
        counts = ", ".join(f"{d}: {(labels == d).sum().item()}" for d in CLASSES)
        print(f"  images per digit: {counts}")


if __name__ == "__main__":
    main()
