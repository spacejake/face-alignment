"""
Differentiable DSNT operations for use in PyTorch computation graphs.
"""

import numpy as np

from functools import reduce
from operator import mul

import torch
import torch.nn.functional


def _normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector
        dtype: Data type of vector elements
        device: Device to store vector on

    Returns:
        The generated vector
    """
    first = -(length - 1) / length
    last = (length - 1) / length
    return torch.linspace(first, last, length, dtype=dtype, device=device)


def _coord_expectation(heatmaps, dim, transform=None):
    """Calculate the coordinate expected value along an axis.

    Args:
        heatmaps: Normalized heatmaps (probabilities)
        dim: Dimension of the coordinate axis
        transform: Coordinate transformation function, defaults to identity

    Returns:
        The coordinate expected value, `E[transform(X)]`
    """

    dim_size = heatmaps.size()[dim]
    own_coords = _normalized_linspace(dim_size, dtype=heatmaps.dtype, device=heatmaps.device)
    if transform:
        own_coords = transform(own_coords)
    summed = heatmaps.view(-1, *heatmaps.size()[2:])
    for i in range(2 - heatmaps.dim(), 0):
        if i != dim:
            summed = summed.sum(i, keepdim=True)
    summed = summed.view(summed.size(0), -1)
    expectations = summed.mul(own_coords.view(-1, own_coords.size(-1))).sum(-1)
    expectations = expectations.view(*heatmaps.size()[:2])
    return expectations


def _coord_variance(heatmaps, dim):
    """Calculate the coordinate variance along an axis.

    Args:
        heatmaps: Normalized heatmaps (probabilities)
        dim: Dimension of the coordinate axis

    Returns:
        The coordinate variance, `Var[X] =  E[(X - E[x])^2]`
    """

    # mu_x = E[X]
    mu_x = _coord_expectation(heatmaps, dim)
    # var_x = E[(X - mu_x)^2]
    var_x = _coord_expectation(heatmaps, dim, transform=lambda x: (x - mu_x) ** 2)

    return var_x


def dsnt(heatmaps):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """

    dim_range = range(-1, 1 - heatmaps.dim(), -1)
    mu = torch.cat([_coord_expectation(heatmaps, dim).unsqueeze(-1) for dim in dim_range], -1)
    return mu


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (Tensor): Predictions (B x L)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            (B x L), defaults to including everything
    """

    if mask is not None:
        assert mask.size() == losses.size(), 'mask must be the same size as losses'
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""

    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)


def wing_losses(actual, target, wing_w=10, wing_e=2):
    """Calculate the average Wing loss for multi-point samples.

    Zhen-Hua Feng, Josef Kittler, Muhammad Awais, Patrik Huber, Xiao-Jun Wu.
    Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks.
    In Proc. CVPR 2018.

    wing(x) = w * ln(1 + |x|/e) if |x| < w
              |x| - C           otherwise

    w = range of non-linearity (-w,w)
    e = limits the curvature of the nonlinear region
    C = w - w ln(1 + w/e), a constant that smoothly links piecewise linear and nonlinear parts

    # Experiments on AFLW dataset, paper shows best parameters are:
    w = 10
    e = 2

    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)

    """
    wing_c = wing_w - wing_w * torch.log(1 + wing_w/wing_e)

    assert actual.size() == target.size(), 'input tensors must have the same size'

    # Calculate Wing loss between actual and target locations
    delta = torch.abs(actual - target)
    if delta < wing_w:
        dist = wing_w * torch.log(1 + delta/wing_e)
    else:
        dist = delta - wing_c

    return dist.sum(-1, keepdim=False)

def euclidean_losses(actual, target):
    """Calculate the average Euclidean loss for multi-point samples.

    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    """

    assert actual.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = actual - target
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt()
    return dist

def hm_losses(actual, target):
    assert actual.size() == target.size(), 'input tensors must have the same size'

    # Calculate MSE between actual and target heatmaps
    diff = actual - target
    dist_sq = diff.pow(2).sum(-1, keepdim=False).sum(-1, keepdim=False)
    dist = dist_sq.sqrt()
    return dist

def make_gauss(means, size, sigma, normalize=True):
    """Draw Gaussians.

    This function is differential with respect to means.

    Note on ordering: `size` expects [..., depth, height, width], whereas
    `means` expects x, y, z, ...

    Args:
        means: coordinates containing the Gaussian means (units: normalized coordinates)
        size: size of the generated images (units: pixels)
        sigma: standard deviation of the Gaussian (units: pixels)
        normalize: when set to True, the returned Gaussians will be normalized
    """

    # Normalize Means, Assume all sizes is greater than 1
    norm_means = means.clone().detach()
    for n, mean in zip(size, norm_means.split(1, -1)):
        mean /= (n - 1)

    norm_means = norm_means * 2 - 1

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [_normalized_linspace(s, dtype=norm_means.dtype, device=norm_means.device)
                   for s in reversed(size)]

    # PDF = exp(-(x - \mu)^2 / (2 \sigma^2))

    # dists <- (x - \mu)^2
    dists = [(x - norm_mean) ** 2 for x, norm_mean in zip(coords_list, norm_means.split(1, -1))]

    # ks <- -1 / (2 \sigma^2)
    stddevs = [2 * sigma / s for s in reversed(size)]
    ks = [-0.5 * (1 / stddev) ** 2 for stddev in stddevs]

    exps = [(dist * k).exp() for k, dist in zip(ks, dists)]

    # Combine dimensions of the Gaussian
    gauss = reduce(mul, [
        reduce(lambda t, d: t.unsqueeze(d), filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, exps)
    ])

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range, gauss) + 1e-24
    gauss = gauss / val_sum
    return gauss


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence):
    ndims = mu_t.size(-1)
    assert heatmaps.dim() == ndims + 2, 'expected heatmaps to be a {}D tensor'.format(ndims + 2)
    assert heatmaps.size()[:-ndims] == mu_t.size()[:-1]

    gauss = make_gauss(mu_t, heatmaps.size()[2:], sigma_t)
    divergences = divergence(heatmaps, gauss, ndims)
    return divergences

def js_loss(heatmaps, target, ndims=2):
    return _js(heatmaps, target, ndims)


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Jensen-Shannon divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in pixels)

    Returns:
        Per-location JS divergences.
    """

    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js).mean(dim=1)


def heatmaps_to_coords(heatmaps, normalize=False):
    xy = dsnt(heatmaps)
    x, y = xy.split(1, -1)
    coords = torch.cat([x, y], -1)

    # Denormalize
    if not normalize:
        coords = (coords + 1) / 2
        dim = heatmaps.shape[-2:]
        for n, coord in zip(dim, coords.split(1, -1)):
            coord *= (n - 1)

    return coords



if __name__ == '__main__':
    from face_alignment.utils import draw_gaussian, draw_gaussianv2
    from skimage.filters import gaussian
    import matplotlib.pyplot as plt
    from random import randint

    n, c, h, w = 2, 2, 256, 256
    vis = True

    # Generate fake heatmap with random keypoint locations
    random_keypoints = []
    for i in range(n):
        batch = []
        for j in range(c):
            kps = [np.random.uniform(0,w), np.random.uniform(0,h)]
            # hm[i, j, kps[1], kps[0]] = 1.
            batch.append([kps[0],kps[1]])
        random_keypoints.append(batch)

    # Put gaussian to peaks
    random_keypoints = np.array(random_keypoints)
    random_keypoints = torch.from_numpy(random_keypoints)

    hm = make_gauss(random_keypoints, (h, w), sigma=2., normalize=True)

    if vis:
        for i in range(n):
            for j in range(c):
                plt.imshow(hm[i, j])
                plt.show()

    keypoints = heatmaps_to_coords(hm, normalize=False)
    random_keypoints = random_keypoints

    print('Original kps: %s' % random_keypoints)
    print('Estimated kps: %s' % keypoints)
    print('Difference: %s' % ((random_keypoints - keypoints) ** 2).sum().item())
