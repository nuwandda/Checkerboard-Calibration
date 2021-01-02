import numpy as np
import cv2
import util
from scipy.optimize import curve_fit


def create_vij(i, j, homography_list):
    util.info("Creating v" + str(i) + str(j) + "...")
    vij = np.zeros((homography_list.shape[0], 6))

    vij[:, 0] = homography_list[:, 0, i] * homography_list[:, 0, j]
    vij[:, 1] = homography_list[:, 0, i] * homography_list[:, 1, j] + homography_list[:, 1, i] * homography_list[:, 0, j]
    vij[:, 2] = homography_list[:, 1, i] * homography_list[:, 1, j]
    vij[:, 3] = homography_list[:, 2, i] * homography_list[:, 0, j] + homography_list[:, 0, i] * homography_list[:, 2, j]
    vij[:, 4] = homography_list[:, 2, i] * homography_list[:, 1, j] + homography_list[:, 1, i] * homography_list[:, 2, j]
    vij[:, 5] = homography_list[:, 2, i] * homography_list[:, 2, j]

    util.info("DONE.")
    return vij

def compute_intrinsics(homographies):
    util.info("Computing camera intrinsics...")
    # Stack homographies
    homography_list = np.zeros((len(homographies), 3, 3))
    for h, H in enumerate(homographies):
        homography_list[h] = H

    # Generate homogeneous equations
    v00 = create_vij(0, 0, homography_list)
    v01 = create_vij(0, 1, homography_list)
    v11 = create_vij(1, 1, homography_list)

    # Therefore, the two fundamental constraints, from a given homoghraphy, can be written ass 2 homogeneous equations in b.
    # V is a 2n x 6 matrix.
    v = np.zeros((2 * len(homographies), 6))
    v[:len(homographies)] = v01
    v[len(homographies):] = v00 - v11

    # The solution to the equation above is well known as the eigenvector of vTv associated with the smallest eigenvalue
    # (equivalently, the right singular vector of v associated with the smallest singular value).
    U, S, V_t = np.linalg.svd(v)
    idx = np.argmin(S)

    # Once b is estimated, we can compute all camera instrinsic matrix A. You can check Appendix B (page 18) for the details.
    # Matrix B is estimated up to a scale factor.
    b0, b1, b2, b3, b4, b5 = V_t[idx]

    v0 = (b1 * b3 - b0 * b4) / (b0 * b2 - b1 * b1)
    lmbda = b5 - (b3 * b3 + v0 * (b1 * b3 - b0 * b4)) / b0
    alpha = np.sqrt(lmbda / b0)
    beta = np.sqrt(lmbda * b0 / (b0 * b2 - b1 * b1))
    gamma = -b1 * alpha * alpha * beta / lmbda
    u0 = gamma * v0 / beta - b3 * alpha * alpha / lmbda

    # Then, create the intrinsic matrix
    A = np.array([[alpha, gamma, u0],
                         [0.,  beta, v0],
                         [0.,    0., 1.]])

    util.info("DONE.\n")
    return A
