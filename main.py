from __future__ import annotations
import functools
from pathlib import Path
import math

import jax
import jax.random as jr
import jax.numpy as jnp
import chex
from scipy import io
from tqdm import trange
import matplotlib.pyplot as plt


def random_crop(
    rng: chex.PRNGKey, x: chex.Array, patch_size: int, flatten: bool = False
) -> chex.Array:
    """Randomly crop a patch from an image.

    Args:
        rng: A PRNG key.
        x: An image array to crop.
        patch_size: Desired size of patches.
        flatten: If True, return a patch as a one-dimensional vector.

    Returns:
        A cropped patch.

    """
    H, W = x.shape
    rng_x, rng_y = jr.split(rng)
    y_offset = jr.randint(rng_y, (), 0, H - patch_size + 1)
    x_offset = jr.randint(rng_x, (), 0, W - patch_size + 1)
    slice_sizes = (patch_size, patch_size)
    x = jax.lax.dynamic_slice(x, (y_offset, x_offset), slice_sizes)
    if flatten:
        x = x.flatten()
    return x


def vis_basis_fns(file: Path, phi: chex.Array, nrows: int = 10) -> None:
    """Visualize the basis functions.

    Args:
        file: Filename to save.
        phi: Basis functions.
        nrows: Number of the grid rows.

    Returns:
        None

    """
    squared_size, num_basis_fns = phi.shape
    size = int(math.sqrt(squared_size))
    assert size * size == squared_size

    fig = plt.figure()
    for j, v in enumerate(phi.reshape(size, size, -1).transpose(2, 0, 1)):
        ax = fig.add_subplot(nrows, math.ceil(num_basis_fns / nrows), j + 1)
        ax.axis("off")
        ax.imshow(v, cmap="gray")
    fig.tight_layout()
    fig.savefig(file)
    plt.close(fig)


def vis_img(file: Path, image: chex.Array) -> None:
    """Visualize the given array as an image.

    Args:
        file: Filename to save.
        image: Image array.

    Returns:
        None

    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis("off")
    ax.imshow(image, cmap="gray")
    fig.savefig(file)
    plt.close(fig)


def train_fn(
    rng: chex.PRNGKey,
    phi: chex.Array,
    images: chex.Array,
    learning_rate: float,
    num_expect_iters: int,
) -> chex.Array:
    """Update the basis functions once.

    Args:
        rng: A PRNG key.
        phi: Basis functions. [crop_size^2, num_basis_fns].
        images: Image arrays. [num_images, height, width].
        learning_rate: Learning rate.
        num_expect_iters: Number of expectation iterations.

    Returns:
        Updated basis functions.

    """
    patch_size = int(math.sqrt(phi.shape[0]))
    assert phi.shape[0] == patch_size**2

    # Patch sampling.
    rng, choice_rng, crop_rng = jr.split(rng, 3)
    x = jr.choice(choice_rng, images)
    x = random_crop(crop_rng, x, patch_size, flatten=True)
    x = (x - x.mean()) / x.std()
    x = x[:, None]

    # E-step w/ ARD
    @jax.jit
    def scan_fn(carry, _):
        z, lam = carry
        grad = phi.T @ (x - phi @ z) - lam[:, None] * z
        hessian = -phi.T @ phi - jnp.diag(lam)
        new_z = z + jnp.linalg.inv(-hessian) @ grad
        z_map = jnp.linalg.inv(phi.T @ phi + jnp.diag(lam)) @ phi.T @ x
        W = -jnp.linalg.inv(hessian)
        new_lam = 1.0 / jnp.diag(W + z_map @ z_map.T)
        return (new_z, new_lam), None

    num_basis_fns = phi.shape[-1]
    z = jnp.zeros([num_basis_fns, 1])
    lam = jnp.ones([num_basis_fns])
    (z, lam), _ = jax.lax.scan(scan_fn, init=(z, lam), xs=jnp.arange(num_expect_iters))

    # M-step
    hessian = -phi.T @ phi - jnp.diag(lam)
    z_map = jnp.linalg.inv(phi.T @ phi + jnp.diag(lam)) @ phi.T @ x
    W = -jnp.linalg.inv(hessian)
    grad_phi = x @ z_map.T - phi @ (W + z_map @ z_map.T)
    new_phi = phi + learning_rate * grad_phi
    new_phi /= jnp.sqrt(jnp.sum(new_phi**2, axis=0, keepdims=True))

    return new_phi


def init_fn(rng: chex.PRNGKey, num_basis_fns: int, patch_size: int) -> chex.Array:
    """Initialize the basis functions.

    Args:
        rng: A PRNG key.
        num_basis_fns: Number of the basis functions.
        patch_size: Patch size.

    Returns:
        Initialized basis functions.

    """
    return jr.normal(rng, (patch_size**2, num_basis_fns))


def cli_main(
    out_dir: str,
    seed: int = 1234,
    num_iters: int = 20000,
    num_expect_iters: int = 3,
    learning_rate: float = 0.01,
    num_basis_fns: int = 100,
    patch_size: int = 12,
    num_rows: int = 10,
) -> None:
    """Main func.

    Args:
        out_dir: Output directory.
        seed: Random seed value.
        num_iters: Number of iterations to update the basis functions.
        num_expect_iters: Number of E-step loops per iteration.
        learning_rate: Learning rate.
        num_basis_fns: Number of the basis functions.
        patch_size: Patch size.
        num_rows: Number of rows to visualize the basis functions.

    Returns:
        None

    """
    # Prepare output directory.
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Prepare variables.
    rng = jr.PRNGKey(seed)
    images = io.loadmat("IMAGES.mat")["IMAGES"]
    images = images.transpose(2, 0, 1)
    fn = jax.jit(
        functools.partial(
            train_fn,
            images=images,
            learning_rate=learning_rate,
            num_expect_iters=num_expect_iters,
        )
    )

    # Homework (1)
    vis_img(out_dir_path / "image.jpg", images[-1])

    # Homework (2)
    rng, crop_rng = jr.split(rng)
    vis_img(out_dir_path / "patch.jpg", random_crop(crop_rng, images[-1], patch_size))

    # Homework (3)
    rng, init_rng = jr.split(rng)
    phi = init_fn(init_rng, num_basis_fns, patch_size)
    for _ in trange(num_iters):
        rng, new_rng = jr.split(rng)
        phi = fn(new_rng, phi)
    vis_basis_fns(out_dir_path / "basis_fns.jpg", phi, num_rows)


if __name__ == "__main__":
    import os
    import fire

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    fire.Fire(cli_main)
