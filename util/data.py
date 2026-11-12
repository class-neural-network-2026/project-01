"""Dataset for the MNIST subset (digits 0, 1, 2, 3, 4)
(provided: do not modify this file).

The data files are created by `prepare_data_mnist.py`:
    data/mnist_train.pt :  100 images per digit ( 500 total), for training
    data/mnist_test.pt  : 1000 images per digit (5000 total), for testing

The only interface is `get_dataset(split)`, where the option `split` selects
the data: "train" for training or "test" for testing. The dataset provides
the images only, without their labels.

Usage (with the `util` folder on sys.path):
    from data import get_dataset

    train_dataset = get_dataset("train")
    test_dataset = get_dataset("test")

    len(train_dataset)                  # number of images
    image = train_dataset[0]
    # image : torch.float32, shape (1, 32, 32), values in [0, 1]

    train_dataset.images                # all the images, shape (N, 1, 32, 32)

The dataset is a torch.utils.data.Dataset: it can be given to a
torch.utils.data.DataLoader to iterate over mini-batches of images, of
shape (B, 1, 32, 32).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

# <project root>/data, independent of the directory the notebook is run from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SPLITS = ("train", "test")


class MNISTDataset(Dataset):
    """Images of the MNIST subset, held in memory.

    Args:
        split: "train" or "test".
        data_dir: folder containing mnist_train.pt and mnist_test.pt.
        transform: optional function applied to each image tensor
            (shape (1, 32, 32), float32, values in [0, 1]) when it is read.

    Each item is an image; the labels are not provided.
    """

    def __init__(self, split="train", data_dir=DATA_DIR, transform=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

        path = Path(data_dir) / f"mnist_{split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run prepare_data_mnist.py first to create it"
            )
        data = torch.load(path, weights_only=True)

        # uint8 (N, 1, 32, 32) in 0..255  ->  float32 (N, 1, 32, 32) in [0, 1]
        self.images = data["images"].float().div(255)
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image


def get_dataset(split="train", data_dir=DATA_DIR, transform=None):
    """Return the training dataset (split="train") or the test dataset
    (split="test"). Each item of the dataset is an image, without label."""
    return MNISTDataset(split, data_dir, transform)


if __name__ == "__main__":
    # Quick check: python util/data.py
    for split in SPLITS:
        dataset = get_dataset(split)
        image = dataset[0]
        print(f"{split}: {len(dataset)} images | "
              f"image {tuple(image.shape)} {image.dtype} "
              f"[{dataset.images.min().item():.1f}, {dataset.images.max().item():.1f}]")
