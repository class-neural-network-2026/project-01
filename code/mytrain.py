"""Training of the VAE: loss function and one epoch of training.

TODO: complete `loss_function` and `train_one_epoch`.
"""

import torch
import torch.nn.functional as F


def loss_function(x, x_hat, mu, logvar):
    """VAE loss (negative ELBO) of a batch, averaged over its images.

        loss  = recon + kl

        recon : reconstruction loss, the binary cross-entropy between x_hat
                and x, summed over the pixels of an image
        kl    : KL divergence KL( q(z|x) || p(z) ) with the prior
                p(z) = N(0, I), summed over the latent dimensions

    Both terms are averaged over the B images of the batch.

    Args:
        x:      images,                  shape (B, 1, 32, 32), values in [0, 1].
        x_hat:  decoded images,          shape (B, 1, 32, 32), values in [0, 1].
        mu:     mean of q(z|x),          shape (B, latent_dim).
        logvar: log-variance of q(z|x),  shape (B, latent_dim).

    Returns:
        (loss, recon, kl): three scalar tensors.
    """
    # TODO
    raise NotImplementedError("mytrain.py: loss_function is not implemented")


def train_one_epoch(model, optimizer, dataloader, device):
    """Train `model` for one epoch, i.e. one pass over `dataloader`.

    For every mini-batch: move the images to `device`, run the model,
    compute the loss with `loss_function`, back-propagate, and update the
    parameters with `optimizer`. Put the model in training mode first.

    Args:
        model:      the VAE, already on `device`.
        optimizer:  the optimizer from myoptim.get_optimizer.
        dataloader: yields batches of images, shape (B, 1, 32, 32), values
                    in [0, 1] (there are no labels).
        device:     torch.device to compute on.

    Returns:
        dict {"loss": ..., "recon": ..., "kl": ...} of Python floats: the
        average of each term per image over the whole epoch.
    """
    # TODO
    raise NotImplementedError("mytrain.py: train_one_epoch is not implemented")
