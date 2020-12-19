import numpy as np
import logging
import inspect


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

def to_inhomogeneous(points):
    new_x = points[0] / points[2]
    new_y = points[1] / points[2]
    new_x = new_x.tolist()
    new_y = new_y.tolist()

    return [new_x[0][0], new_y[0][0]]

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
    std_x = np.sqrt(2 / var_x)
    std_y = np.sqrt(2 / var_y)
    transformation = np.array([[std_x, 0, -std_x * mean_x],
                               [0, std_y, -std_y * mean_y],
                               [0, 0, 1]])

    return transformation


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
