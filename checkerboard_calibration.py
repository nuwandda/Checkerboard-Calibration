import numpy as np
import cv2
import glob
import logging


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
            cv2.imshow('img', img)
            cv2.waitKey(500)
            count += 1

    cv2.destroyAllWindows()

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


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    world_coordinates, image_coordinates = find_corners()
    converted_image_coordinates = convert_to_2d(image_coordinates)
    converted_world_coordinates = convert_to_3d(world_coordinates)
    normalized_image_coors, t = normalize_points_2d(converted_image_coordinates)
    normalized_world_coors, u = normalize_points_3d(converted_world_coordinates)


if __name__ == "__main__":
    main()
