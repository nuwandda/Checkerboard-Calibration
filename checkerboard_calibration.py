import numpy as np
import cv2
import glob
import logging
from scipy import linalg


def find_corners():
    logging.info('find_corners function has started...')
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checkerboard = (6, 9)
    # Prepare coordinates
    object_points = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    object_points[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    # Lists to store coordinates from both world and image planes
    world_coordinates = []
    image_coordinates = []

    images = glob.glob('images/*.jpg')
    logging.info('Finding chessboard corners...')
    count = 1
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret:
            world_coordinates.append(object_points)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), termination_criteria)
            image_coordinates.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
            cv2.imwrite('images/pattern_' + str(count) + '.png', img)
            # cv2.imshow('img', img)
            #3cv2.waitKey(500)
            count += 1

    # cv2.destroyAllWindows()

    return world_coordinates, image_coordinates


def convert_to_2d(points):
    logging.info('Converting image points to 2D...')
    converted_list = []
    for group in points:
        for point in group:
            converted_list.append(point[0].tolist())

    return converted_list


def convert_to_3d(points):
    logging.info('Converting world points to 3D...')
    converted_list = []
    for group in points:
        for point in group:
            for p in point:
                converted_list.append(p.tolist())

    return converted_list


def normalize_points_2d(points):
    logging.info('Normalizing 2D points using similarity transformation...')
    points = np.array(points)
    mean, std = np.mean(points, 0), np.std(points)

    # Create similarity transformation matrix
    # Set centroid as origin
    transformation = np.array([[std / np.sqrt(2), 0, mean[0]],
                               [0, std / np.sqrt(2), mean[1]],
                               [0, 0, 1]])

    # Apply transformation on points
    transformation = np.linalg.inv(transformation)
    points = np.dot(transformation, np.concatenate((points.T, np.ones((1, points.shape[0])))))
    points = points[0:2].T

    return points, transformation


def normalize_points_3d(points):
    logging.info('Normalizing 3D points using similarity transformation...')
    points = np.array(points)
    mean, std = np.mean(points, 0), np.std(points)

    # Create similarity transformation matrix
    # Set centroid as origin
    transformation = np.array([[std / np.sqrt(3), 0, 0, mean[0]],
                               [0, std / np.sqrt(3), 0, mean[1]],
                               [0, 0, std / np.sqrt(3), mean[2]],
                               [0, 0, 0, 1]])

    # Apply transformation on points
    transformation = np.linalg.inv(transformation)
    points = np.dot(transformation, np.concatenate((points.T, np.ones((1, points.shape[0])))))
    points = points[0:3].T

    return points, transformation


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


def create_camera_matrix(image_points, world_points):
    logging.info('Creating camera matrix...')
    a_list = []
    xyz = np.asarray(world_points)
    ab = np.asarray(image_points)
    # Normalize points with regarding dimensions and get similarity transformations
    normalized_image_coors, t = normalize_points_2d(image_points)
    normalized_world_coors, u = normalize_points_3d(world_points)
    for i in range(len(normalized_image_coors)):
        # In Zhang's calibration method, the world coordinate system is placed on the checkerboard plane,
        # and the checkerboard plane is set to the plane with z = 0.
        # But in this project, we will use exact points
        # First, convert each points into homogeneous coordinates
        point1 = np.matrix([normalized_image_coors[i][0], normalized_image_coors[i][1], 1])
        point2 = np.matrix([normalized_world_coors[i][0], normalized_world_coors[i][1], 0, 1])
        a, b = point1.item(0), point1.item(1)
        x, y, z = point2.item(0), point2.item(1), point2.item(2)

        # Form the 2n x 12 matrix A by stacking the equations generated by each correspondence. Write 'p' for the vector
        # containing the entries of the matrix P.
        a_row1 = [x, y, z, 1, 0, 0, 0, 0, -a * x, -a * y, -a * z, -a]
        a_row2 = [0, 0, 0, 0, x, y, z, 1, -b * x, -b * y, -b * z, -b]

        # Assemble the 2n x 9 matrices Ai into a single matrix A.
        a_list.append(a_row1)
        a_list.append(a_row2)

    a_matrix = np.matrix(a_list)

    # A solution of Ap = 0, subject to ||p|| = 1, is obtained from the unit singular vector of A corresponding to the
    # smallest singular value.
    U, S, V = np.linalg.svd(a_matrix)

    # The parameters are in the last line of Vh and normalize them
    H = np.reshape(V[8], (3, 4))
    H = (1 / H.item(8)) * H

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot(np.dot(np.linalg.pinv(t), H), u)
    #L = H / H[-1, -1]
    # L = H.flatten()

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    # uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    # uv2 = uv2 / uv2[2, :]
    # Mean distance:
    # err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - ab)**2, 1)))

    return H


def decompose_camera(P):
    # Define H as the first three columns of P.
    H = P[:, 0:3]
    print("First three columns of P")
    print(H)
    print("---")
    # K and R comes from an RQ decomposition.
    k, r = linalg.rq(H)
    print("K")
    print(k)
    print("---")
    print("R")
    print(r)
    print("-----------")
    H = P.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(k)
    L = 1 / np.linalg.norm(np.dot(h1, K_inv))
    r1 = L * np.dot(h1, K_inv)
    r2 = L * np.dot(h2, K_inv)
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))
    print(T)
    print("---")
    print(R)


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    world_coordinates, image_coordinates = find_corners()
    converted_image_coordinates = convert_to_2d(image_coordinates)
    converted_world_coordinates = convert_to_3d(world_coordinates)
    L = create_camera_matrix(converted_image_coordinates, converted_world_coordinates)
    print("Homography")
    print(L)
    print("---")
    xyz = np.asarray(converted_world_coordinates)
    ab = np.asarray(converted_image_coordinates)
    # h_cv = cv2.findHomography(xyz, ab)
    decompose_camera(L)


if __name__ == "__main__":
    main()
