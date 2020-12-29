import numpy as np
import cv2
import util


def estimate_radial_distortion(obj_points, img_points, K, extrinsics):
    util.info("Estimating radial distortion by alternation...")
    M = len(img_points)
    N = obj_points[0].shape[0]

    model = util.to_homogeneous_3d_multiple_points(obj_points[0])

    u_c, v_c = K[0,2], K[1,2]

    # Form radius vector
    r = np.zeros(2 * M * N)
    for e, E in enumerate(extrinsics):
        normalized_projection = np.dot(model, E.T)
        normalized_projection = util.to_inhomogeneous_multiple_points(normalized_projection)

        x_normalized_proj, y_normalized_proj = normalized_projection[:, 0], normalized_projection[:, 1]
        r_i = np.sqrt(x_normalized_proj**2 + y_normalized_proj**2)
        r[e*N:(e+1)*N] = r_i
    r[M*N:] = r[:M*N]

    # Form observation vector
    obs = np.zeros(2 * M * N)
    u_data, v_data = np.zeros(M * N), np.zeros(M * N)
    for d, data in enumerate(img_points):
        u_i, v_i = data[:, 0][:, 0], data[:, 0][:, 1]
        u_data[d*N:(d+1)*N] = u_i
        v_data[d*N:(d+1)*N] = v_i
    obs[:M*N] = u_data
    obs[M*N:] = v_data

    # Form prediction vector
    pred = np.zeros(2 * M * N)
    pred_centered = np.zeros(2 * M * N)
    u_pred, v_pred = np.zeros(M * N), np.zeros(M * N)
    for e, E in enumerate(extrinsics):
        P = np.dot(K, E)
        projection = np.dot(model, P.T)
        projection = util.to_inhomogeneous_multiple_points(projection)
        u_pred_i = projection[:, 0]
        v_pred_i = projection[:, 1]

        u_pred[e*N:(e+1)*N] = u_pred_i
        v_pred[e*N:(e+1)*N] = v_pred_i
    pred[:M*N] = u_pred
    pred[M*N:] = v_pred
    pred_centered[:M*N] = u_pred - u_c
    pred_centered[M*N:] = v_pred - v_c

    # Form distortion coefficient constraint matrix
    D = np.zeros((2 * M * N, 2))
    D[:, 0] = pred_centered * r**2
    D[:, 1] = pred_centered * r**4

    # Form values (difference between sensor observations and predictions)
    b = obs - pred

    # Use pseudoinverse technique to compute least squares solution for distortion coefficients
    D_inv = np.linalg.pinv(D)
    k = np.dot(D_inv, b)
    util.info("DONE.")

    return k
