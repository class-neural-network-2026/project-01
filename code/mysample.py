"""Generation of new images with a trained VAE.

TODO: complete `sample`.
"""

import torch


def sample(model, num_samples, device):
    """Generate new images: draw z from the prior p(z) = N(0, I) and decode.

    The model is not updated: put it in evaluation mode and do not track
    gradients (torch.no_grad).

    Args:
        model:       the VAE, already on `device`.
        num_samples: number of images to generate.
        device:      torch.device to compute on.

    Returns:
        generated images, shape (num_samples, 1, 32, 32), values in [0, 1].
    """
    # TODO
    raise NotImplementedError("mysample.py: sample is not implemented")
