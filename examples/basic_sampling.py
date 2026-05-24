import math

import matplotlib.pyplot as plt
import torch

import sqclouds as sq

def set_axes_equal(ax) -> None:
    """Make a 3D matplotlib plot use equal scale on all axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = sum(x_limits) / 2.0
    y_middle = sum(y_limits) / 2.0
    z_middle = sum(z_limits) / 2.0

    ax.set_xlim3d(x_middle - max_range / 2.0, x_middle + max_range / 2.0)
    ax.set_ylim3d(y_middle - max_range / 2.0, y_middle + max_range / 2.0)
    ax.set_zlim3d(z_middle - max_range / 2.0, z_middle + max_range / 2.0)


def main() -> None:
    a_x = 1.0
    a_y = 0.5
    a_z = 0.35
    eps_xy = 0.6
    eps_z = 1.2

    n_phi_quad = 32
    n_theta_quad = 32

    points_base = sq.sample_superellipsoid(
        a_x,
        a_y,
        a_z,
        eps_xy,
        eps_z,
        n_phi_quad,
        n_theta_quad,
    )

    rotation = sq.rotation_matrix_z(math.radians(45.0))
    translation = torch.tensor([2.0, 0.5, 0.25], dtype=torch.float32)

    points_transformed = sq.sample_superellipsoid(
        a_x,
        a_y,
        a_z,
        eps_xy,
        eps_z,
        n_phi_quad,
        n_theta_quad,
        rotation=rotation,
        translation=translation,
    )

    print("base shape:", points_base.shape)
    print("transformed shape:", points_transformed.shape)

    base = points_base.detach().cpu()
    transformed = points_transformed.detach().cpu()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(base[:, 0], base[:, 1], base[:, 2], s=2, label="base")
    ax.scatter(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        s=2,
        label="rotated + translated",
    )

    ax.set_title("Superellipsoid sampling with rotation and translation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    set_axes_equal(ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()