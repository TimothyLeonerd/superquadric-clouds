import sqclouds as sq


def main() -> None:
    a_x = 1.0
    a_y = 1.0
    a_z = 1.0
    eps_xy = 1.0
    eps_z = 1.0

    n_phi_quad = 4
    n_theta_quad = 2

    points = sq.sample_superellipsoid(
        a_x,
        a_y,
        a_z,
        eps_xy,
        eps_z,
        n_phi_quad,
        n_theta_quad,
    )

    print(points)
    print("shape:", points.shape)


if __name__ == "__main__":
    main()