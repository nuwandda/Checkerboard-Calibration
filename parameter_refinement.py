import math
import numpy as np
from scipy import optimize as opt
import util


def get_single_project_coordinates(A, W, k, coor):
    single_coor = np.array([coor[0], coor[1], 0, 1])

    coor_norm = np.dot(W, single_coor)
    coor_norm /= coor_norm[-1]

    r = np.linalg.norm(coor_norm)

    uv = np.dot(np.dot(A, W), single_coor)
    uv /= uv[-1]

    u0 = uv[0]
    v0 = uv[1]

    uc = A[0, 2]
    vc = A[1, 2]

    u = u0 + (u0 - uc) * r**2 * k[0] + (u0 - uc) * r**4 * k[1]
    v = v0 + (v0 - vc) * r**2 * k[0] + (v0 - vc) * r**4 * k[1]

    return np.array([u, v])

def compose_parameter_vector(A, k, W):
    alpha = np.array([A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2], k[0], k[1]])
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:, :3], (W[i])[:, 3]

        rotation_vector = to_rodrigues_vector(R)

        w = np.append(rotation_vector, t)
        P = np.append(P, w)
    return P

def decompose_parameter_vector(parameters):
    [alpha, beta, gamma, uc, vc, k0, k1] = parameters[0:7]
    A = np.array([[alpha, gamma, uc],
                  [0, beta, vc],
                  [0, 0, 1]])
    k = np.array([k0, k1])
    W = []
    M = (len(parameters) - 7) // 6

    for i in range(M):
        m = 7 + 6 * i
        rotation_vector = parameters[m:m+3]
        t = (parameters[m+3:m+6]).reshape(3, -1)

        R = to_rotation_matrix(rotation_vector)

        w = np.concatenate((R, t), axis=1)
        W.append(w)

    W = np.array(W)
    return A, k, W

def value_function(parameters, W, obj_points, img_points):
    # The function calculates the value of the optimization model for the target points in X and the model parameters P. 
    # The Jacobian function calculates the associated Jacobian matrix.
    M = (len(parameters) - 7) // 6
    N = len(obj_points[0])
    A = np.array([
        [parameters[0], parameters[2], parameters[3]],
        [0, parameters[1], parameters[4]],
        [0, 0, 1]
    ])
    Y = np.array([])

    for i in range(M):
        m = 7 + 6 * i
        w = parameters[m:m + 6]
        W_temp = W[i]

        for j in range(N):
            Y = np.append(Y, get_single_project_coordinates(A, W_temp, np.array([parameters[5], parameters[6]]), (obj_points[i])[j]))

    error_Y  =  np.array(img_points).reshape(-1) - Y

    return error_Y

def jacobian_function(parameters, extrinsics, obj_points, img_points):
    # Returns the Jacobian matrixof size 2M N×K(withK= 7 + 6M).
    M = (len(parameters) - 7) // 6
    N = len(obj_points[0])
    K = len(parameters)
    A = np.array([
        [parameters[0], parameters[2], parameters[3]],
        [0, parameters[1], parameters[4]],
        [0, 0, 1]
    ])

    res = np.array([])

    for i in range(M):
        m = 7 + 6 * i

        w = parameters[m:m + 6]
        R = to_rotation_matrix(w[:3])
        t = w[3:].reshape(3, 1)
        W = np.concatenate((R, t), axis=1)

        for j in range(N):
            res = np.append(res, get_single_project_coordinates(A, W, np.array([parameters[5], parameters[6]]), (obj_points[i])[j]))

    J = np.zeros((K, 2 * M * N))
    for k in range(K):
        J[k] = np.gradient(res, parameters[k])

    return J.T

def to_rodrigues_vector(rotation_matrix):
    # Returns the associatedRodrigues rotation vector
    p = 0.5 * np.array([[rotation_matrix[2, 1] - rotation_matrix[1, 2]],
                        [rotation_matrix[0, 2] - rotation_matrix[2, 0]],
                        [rotation_matrix[1, 0] - rotation_matrix[0, 1]]])
    c = 0.5 * (np.trace(rotation_matrix) - 1)

    if np.linalg.norm(p) == 0:
        if c == 1:
            rotation_vector = np.array([0, 0, 0])
        elif c == -1:
            R_plus = rotation_matrix + np.eye(3, dtype='float')

            norm_array = np.array([np.linalg.norm(R_plus[:, 0]),
                                   np.linalg.norm(R_plus[:, 1]),
                                   np.linalg.norm(R_plus[:, 2])])
            v = R_plus[:, np.where(norm_array == max(norm_array))]
            u = v / np.linalg.norm(v)

            if u[0] < 0 or (u[0] == 0 and u[1] < 0) or (u[0] == u[1] and u[0] == 0 and u[2] < 0):
                u = -u
            rotation_vector = math.pi * u
        else:
            rotation_vector = []
    else:
        u = p / np.linalg.norm(p)
        theata = math.atan2(np.linalg.norm(p), c)
        rotation_vector = theata * u

    return rotation_vector

def to_rotation_matrix(rotation_vector):
    # Returns the asso-ciated rotation matrix
    theta = np.linalg.norm(rotation_vector)
    unit_vector = rotation_vector / theta

    W = np.array([[0, -unit_vector[2], unit_vector[1]],
                  [unit_vector[2], 0, -unit_vector[0]],
                  [-unit_vector[1], unit_vector[0], 0]])
    R = np.eye(3, dtype='float') + W * math.sin(theta) + np.dot(W, W) * (1 - math.cos(theta))

    return R

def refine(A, k, W, obj_points, img_points):
    # Overall, nonlinear refinement.
    # Goal: find the intrinsic and extrinsic parameters that minimize E=∑M−1,i=0 ∑N−1,j=0 ‖ ̇ui,j−P(a,wi,Xj)‖2.
    # The overall refinement step is to determinethe camera parametersaand the view parameters w0, . . . ,wM−1 
    # that minimize the total projection error E
    util.info("Refining all parameters...")
    P_init = compose_parameter_vector(A, k, W)

    X_double = np.zeros((2 * len(obj_points) * len(obj_points[0]), 2))
    Y = np.zeros((2 * len(obj_points) * len(obj_points[0])))

    M = len(obj_points)
    N = len(obj_points[0])
    for i in range(M):
        for j in range(N):
            X_double[(i * N + j) * 2] = (obj_points[i])[j]
            X_double[(i * N + j) * 2 + 1] = (obj_points[i])[j]
            Y[(i * N + j) * 2] = (img_points[i])[j, 0][0]
            Y[(i * N + j) * 2 + 1] = (img_points[i])[j, 0][1]

    util.info("Nonlinear refinement with LM has started.")
    P = opt.leastsq(value_function,
                    P_init,
                    args=(W, obj_points, img_points),
                    Dfun=jacobian_function)[0]
    util.info("DONE.")

    error = value_function(P, W, obj_points, img_points)
    radial_error = [np.sqrt(error[2 * i]**2 + error[2 * i + 1]**2) for i in range(len(error) // 2)]

    util.info("Error: \n" + str(np.max(radial_error)))
    util.info("DONE.")

    return decompose_parameter_vector(P)
