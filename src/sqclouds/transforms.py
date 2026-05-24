from __future__ import annotations

import math
import torch


def rotation_matrix_x(
    angle: float,
    *,
    degrees: bool = False,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    if degrees:
        angle = math.radians(angle)

    c = math.cos(angle)
    s = math.sin(angle)

    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        device=device,
        dtype=dtype,
    )


def rotation_matrix_y(
    angle: float,
    *,
    degrees: bool = False,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    if degrees:
        angle = math.radians(angle)

    c = math.cos(angle)
    s = math.sin(angle)

    return torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        device=device,
        dtype=dtype,
    )


def rotation_matrix_z(
    angle: float,
    *,
    degrees: bool = False,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    if degrees:
        angle = math.radians(angle)

    c = math.cos(angle)
    s = math.sin(angle)

    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def transform_points(
    points: torch.Tensor,
    *,
    rotation: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a rigid transform to points.

    Args:
        points: Tensor of shape [N, 3], where N is the number of points.
        rotation: Optional rotation matrix of shape [3, 3].
        translation: Optional translation vector of shape [3].

    Returns:
        Tensor of shape [N, 3].
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {tuple(points.shape)}")

    out = points

    if rotation is not None:
        rotation = torch.as_tensor(rotation, device=points.device, dtype=points.dtype)

        if rotation.shape != (3, 3):
            raise ValueError(f"Expected rotation with shape [3, 3], got {tuple(rotation.shape)}")

        out = out @ rotation.T

    if translation is not None:
        translation = torch.as_tensor(translation, device=points.device, dtype=points.dtype)

        if translation.shape != (3,):
            raise ValueError(
                f"Expected translation with shape [3], got {tuple(translation.shape)}"
            )

        out = out + translation

    return out