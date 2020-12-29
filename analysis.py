import matplotlib.pyplot as plt
import util


def plot_differences(img_points, obj_points, homography, filename):
    util.info("Plotting difference of object points and projected points...")
    correspondences = util.get_correspondences(img_points, homography)

    x1 = obj_points[:, 0].tolist()
    y1 = obj_points[:, 1].tolist()

    plt.plot(x1, y1, label="Object Points")

    x2 = util.column(correspondences, 0)
    y2 = util.column(correspondences, 1)

    plt.plot(x2, y2, label="Projected Points")
    plt.xlabel('X Axis')

    plt.ylabel('Y Axis')

    plt.title('Comparison of object points and projected points')

    plt.legend()

    util.info("Writing graph...")
    plt.savefig("graphs/differences_" + filename + ".png")
    plt.clf()
    util.info("DONE.\n")
