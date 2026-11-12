"""Optimization settings: batch size, number of epochs, optimizer.

TODO: choose BATCH_SIZE and NUM_EPOCHS, and complete `get_optimizer`.
"""

import torch

# Used by the notebook. The values below are only a starting point.
BATCH_SIZE = 64     # number of images in a mini-batch
NUM_EPOCHS = 100    # number of passes over the training data


def get_optimizer(model):
    """Create the optimizer that updates the parameters of `model`.

    The choice of the optimizer and of its settings (learning rate, ...) is
    yours.

    Args:
        model: the VAE defined in mymodel.py.

    Returns:
        a torch.optim.Optimizer.
    """
    # TODO
    raise NotImplementedError("myoptim.py: get_optimizer is not implemented")
