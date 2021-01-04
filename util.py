import numpy as np
import logging
import inspect
import scipy.linalg
import math


def info(msg):
    logger = logging.getLogger(__name__)
    frame, filename, line_number, function_name, lines, index = inspect.getouterframes(
        inspect.currentframe())[1]
    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logger.info('{i}{f}(): {m}'.format(
        f=function_name,
        i=' '*int(indentation_level / 4),
        m=msg
    ))

def to_homogeneous(points, points_type):
    if points_type == 0:
        return np.matrix([points[0][0], points[0][1], 1])
    else:
        return np.matrix([points[0], points[1], 1])

def to_homogeneous_3d(points, points_type):
    if points_type == 0:
        return np.matrix([points[0][0], points[0][1], 0, 1])
    else:
        return np.matrix([points[0], points[1], 0, 1])

def to_inhomogeneous(points):
    new_x = points[0] / points[2]
    new_y = points[1] / points[2]
    new_x = new_x.tolist()
    new_y = new_y.tolist()

    return [new_x[0][0], new_y[0][0]]

def to_homogeneous_multiple_points(A):
    A = np.atleast_2d(A)

    N = A.shape[0]
    A_hom = np.hstack((A, np.ones((N,1))))

    return A_hom

def to_inhomogeneous_multiple_points(A):
    A = np.atleast_2d(A)

    N = A.shape[0]
    A /= A[:,-1][:, np.newaxis]
    A_inhom = A[:,:-1]

    return A_inhom

def to_homogeneous_3d_multiple_points(A):
    if A.ndim != 2 or A.shape[-1] != 2:
        raise ValueError('Stacked vectors must be 2D inhomogeneous')

    N = A.shape[0]
    A_3d = np.hstack((A, np.zeros((N,1))))
    A_3d_hom = to_homogeneous_multiple_points(A_3d)

    return A_3d_hom

def get_transformation_matrix(points, points_type):
    if points_type == 0:
        x, y = points[:, 0][:, 0], points[:, 0][:, 1]
        info("Normalizing image points using similarity transformation...")
    else:
        x, y = points[:, 0], points[:, 1]
        info("Normalizing object points using similarity transformation...")

    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    # Create similarity transformation matrix
    # Set centroid as origin
    std_x = np.sqrt(2. / var_x)
    std_y = np.sqrt(2. / var_y)
    transformation = np.array([[std_x, 0., -std_x * mean_x],
                               [0., std_y, -std_y * mean_y],
                               [0., 0., 1.]])

    return transformation

def get_correspondences(img_points, homography):
    info("Calculating correspondences with homography...")
    correspondences = []
    for point in img_points:
        homogeneous_point = to_homogeneous(point, 0)
        correspondence = np.matmul(homography, homogeneous_point.T)
        correspondences.append(to_inhomogeneous(correspondence))

    info("DONE.")
    return correspondences

def get_normalization_matrix_3d(points):
    logging.info('Normalizing 3D points using similarity transformation...')
    points = np.array(points)
    mean, std = np.mean(points, 0), np.std(points)

    # Create similarity transformation matrix
    # Set centroid as origin
    transformation = np.array([[std / np.sqrt(3), 0, 0, mean[0]],
                               [0, std / np.sqrt(3), 0, mean[1]],
                               [0, 0, std / np.sqrt(3), mean[2]],
                               [0, 0, 0, 1]])

    return transformation

def geometric_error(correspondence, h):
    # Thus, geometric distance is related to, but not quite the same as, algebraic distance.
    # Note, though, that if z coordinates equal 1, then the two distances are identical.
    logging.info('Calculating geometric distance...')
    point1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimated_point2 = np.dot(h, point1)
    estimated_point2 = (1 / estimated_point2.item(2)) * estimated_point2

    point2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = point2 - estimated_point2
    distance = np.linalg.norm(error)
    return distance / np.dot(point1.item(2), point2.item(2))

def column(matrix, i):
    return [row[i] for row in matrix]

def row(matrix, i):
    pass

def reorthogonalize(R):
    U, S, V_t = np.linalg.svd(R)
    new_R = np.dot(U, V_t)
    
    return new_R

def distort(k, normalized_proj):
    x, y = normalized_proj[:, 0], normalized_proj[:, 1]

    # Calculate radii
    r = np.sqrt(x**2 + y**2)

    k0, k1 = k

    # Calculate distortion effects
    D = k0 * r**2 + k1 * r**4
    
    # Calculate distorted normalized projection values
    x_prime = x * (1. + D)
    y_prime = y * (1. + D)

    distorted_proj = np.hstack((x_prime[:, np.newaxis], y_prime[:, np.newaxis]))

    return distorted_proj

def project(K, k, E, model):
    model_hom = to_homogeneous_3d_multiple_points(model)

    normalized_proj = np.dot(model_hom, E.T)
    normalized_proj = to_inhomogeneous_multiple_points(normalized_proj)

    distorted_proj = distort(k, normalized_proj)
    distorted_proj_hom = to_homogeneous_multiple_points(distorted_proj)

    sensor_proj = np.dot(distorted_proj_hom, K[:-1].T)

    return sensor_proj

def get_camera_matrix(A, extrinsic):
    return np.matmul(A, extrinsic)

def decompose(pmat):
    M = pmat[:,:3]
    K,R = my_rq(M)
    K = K/K[2,2] # Normalize intrinsic parameter matrix
    C_ = pmat2cam_center(pmat)
    t = np.dot( -R, C_)
    Rt = np.hstack((R, t ))

    return dict( intrinsic=K,
                 rotation=R,
                 cam_center=C_,
                 t=t,
                 extrinsic=Rt)

def to_opengl_projection(K, x0, y0, width, height, znear, zfar, direction=None):
    znear = float(znear)
    zfar = float(zfar)
    depth = zfar - znear
    q = -(zfar + znear) / depth
    qn = -2 * (zfar * znear) / depth

    if direction=='y up':
        proj = np.array([[ 2*K[0,0]/width, -2*K[0,1]/width, (-2*K[0,2]+width+2*x0)/width, 0 ],
                         [  0,             -2*K[1,1]/height,(-2*K[1,2]+height+2*y0)/height, 0],
                         [0,0,q,qn],  # This row is standard glPerspective and sets near and far planes.
                         [0,0,-1,0]]) # This row is also standard glPerspective.
    else:
        assert direction=='y down'
        proj = np.array([[ 2*K[0,0]/width, -2*K[0,1]/width, (-2*K[0,2]+width+2*x0)/width, 0 ],
                         [  0,              2*K[1,1]/height,( 2*K[1,2]-height+2*y0)/height, 0],
                         [0,0,q,qn],  # This row is standard glPerspective and sets near and far planes.
                         [0,0,-1,0]]) # This row is also standard glPerspective.
    return proj

def my_rq(M):
    # RQ decomposition, ensures diagonal of R is positive
    R,K = scipy.linalg.rq(M)
    n = R.shape[0]
    for i in range(n):
        if R[i,i]<0:
            R[:,i] = -R[:,i]
            K[i,:] = -K[i,:]

    return R,K

def pmat2cam_center(P):
    # See Hartley & Zisserman (2003) p. 163
    assert P.shape == (3,4)
    determinant = np.linalg.det

    # camera center
    X = determinant( [ P[:,1], P[:,2], P[:,3] ] )
    Y = -determinant( [ P[:,0], P[:,2], P[:,3] ] )
    Z = determinant( [ P[:,0], P[:,1], P[:,3] ] )
    T = -determinant( [ P[:,0], P[:,1], P[:,2] ] )

    C_ = np.transpose(np.array( [[ X/T, Y/T, Z/T ]] ))

    return C_

def isRotationMatrix(R):
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

