import math

import torch


def mirror_octant_points(points: torch.Tensor) -> torch.Tensor:
    """
    Mirror points from one octant to all eight octants.

    Args:
        points: Tensor of shape [N, 3], where N is the number of points.

    Returns:
        Tensor of shape [8 * N, 3].
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {tuple(points.shape)}")

    signs = torch.tensor(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [-1, -1, -1],
        ],
        device=points.device,
        dtype=points.dtype,
    )

    mirrored = points[:, None, :] * signs[None, :, :]
    return mirrored.reshape(-1, 3)


def _sample_superellipsoid_positive_octant(
    a_x: float | torch.Tensor,
    a_y: float | torch.Tensor,
    a_z: float | torch.Tensor,
    eps_xy: float | torch.Tensor,
    eps_z: float | torch.Tensor,
    n_phi_quad: int,
    n_theta_quad: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample the positive octant of a superellipsoid.

    Args:
        a_x, a_y, a_z: Axis scales.
        eps_xy: Exponent controlling the xy cross-section.
        eps_z: Exponent controlling the elevation/z direction.
        n_phi_quad: Number of azimuth samples in one quadrant.
        n_theta_quad: Number of elevation samples in one quadrant.
        device: PyTorch device.
        dtype: PyTorch dtype.

    Returns:
        Tensor of shape [N, 3], where N = n_phi_quad * n_theta_quad.
    """
    d_phi = math.pi / (2.0 * n_phi_quad)
    d_theta = math.pi / (2.0 * n_theta_quad)

    phi = torch.linspace(
        d_phi / 2.0,
        math.pi / 2.0 - d_phi / 2.0,
        n_phi_quad,
        device=device,
        dtype=dtype,
    )
    theta = torch.linspace(
        d_theta / 2.0,
        math.pi / 2.0 - d_theta / 2.0,
        n_theta_quad,
        device=device,
        dtype=dtype,
    )

    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")

    a_x = torch.as_tensor(a_x, device=device, dtype=dtype)
    a_y = torch.as_tensor(a_y, device=device, dtype=dtype)
    a_z = torch.as_tensor(a_z, device=device, dtype=dtype)
    eps_xy = torch.as_tensor(eps_xy, device=device, dtype=dtype)
    eps_z = torch.as_tensor(eps_z, device=device, dtype=dtype)

    cos_theta_eps = torch.cos(theta_grid).pow(eps_z)

    x = a_x * cos_theta_eps * torch.cos(phi_grid).pow(eps_xy)
    y = a_y * cos_theta_eps * torch.sin(phi_grid).pow(eps_xy)
    z = a_z * torch.sin(theta_grid).pow(eps_z)

    return torch.stack(
        [x.reshape(-1), y.reshape(-1), z.reshape(-1)],
        dim=-1,
    )


def sample_superellipsoid(
    a_x: float | torch.Tensor,
    a_y: float | torch.Tensor,
    a_z: float | torch.Tensor,
    eps_xy: float | torch.Tensor,
    eps_z: float | torch.Tensor,
    n_phi_quad: int,
    n_theta_quad: int,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample a superellipsoid point cloud using positive-octant midpoint sampling
    and reflection symmetry.

    This avoids duplicate points on coordinate-plane seams by sampling only
    the interior of the positive octant and mirroring the result.

    Args:
        a_x, a_y, a_z: Axis scales.
        eps_xy: Exponent controlling the xy cross-section.
        eps_z: Exponent controlling the elevation/z direction.
        n_phi_quad: Number of azimuth samples in one quadrant.
        n_theta_quad: Number of elevation samples in one quadrant.
        device: PyTorch device.
        dtype: PyTorch dtype.

    Returns:
        Tensor of shape [8 * n_phi_quad * n_theta_quad, 3].
    """
    positive_octant = _sample_superellipsoid_positive_octant(
        a_x,
        a_y,
        a_z,
        eps_xy,
        eps_z,
        n_phi_quad,
        n_theta_quad,
        device=device,
        dtype=dtype,
    )
    return mirror_octant_points(positive_octant)