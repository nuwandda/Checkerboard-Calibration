import corner_detection as cd
import homography_operations as ho
import intrinsic_estimation as intr
import extrinsic_estimation as extr
import distortion_estimation as de
import parameter_refinement as pr
import visualize
import analysis
import logging
import util
import pathlib


def main():
    pathlib.Path('graphs/').mkdir(parents=True, exist_ok=True)
    obj_points, img_points, img_shapes, img_names = cd.find_corners()

    refined_homographies = []
    for index in range(len(img_points)):
        util.info("Image Count: " + str(index + 1))
        h = ho.create_homography(img_points[index], obj_points[index])
        util.info("Homography:\n" + str(h) + "\n")
        h = ho.refine_homography(h, img_points[index], obj_points[index])
        util.info("Refined Homography:\n" + str(h) + "\n")
        refined_homographies.append(h)
        # analysis.plot_differences(img_points[index], obj_points[index], h, str(index + 1))
    
    A = intr.compute_intrinsics(refined_homographies)
    util.info("Camera Intrinsics:\n" + str(A) + "\n")

    # Once A is known, the extrinsic parameters for each image is readily computed.
    extrinsics = []
    for h_index in range(len(refined_homographies)):
        E = extr.compute_extrinsics(A, refined_homographies[h_index])
        util.info("Camera Extrinsic Matrix For Image-" + str(h_index + 1) + ":\n" + str(E) + "\n")
        extrinsics.append(E)

    # As the radial distortion is expected to be small, one would expect to estimate the other five intrinsic parameters,
    # using the Maximum likelihood estimation, reasonable well by simply ignoring distortion. One strategy is then to estimate
    # k1 and k2 after having estimated the other parameters, which will give us the ideal pixel coordinates.
    k = de.estimate_radial_distortion(obj_points, img_points, A, extrinsics)
    util.info("Radial Distortion: \n" + str(k) + "\n")

    K_opt, k_opt, extrinsics_opt  = pr.refine(A, k, extrinsics, obj_points, img_points)
    util.info("Parameters:")
    print('\t   Focal Length: [ {:.5f}  {:.5f} ]'.format(K_opt[0,0], K_opt[1,1]))
    print('\tPrincipal Point: [ {:.5f}  {:.5f} ]'.format(K_opt[0,2], K_opt[1,2]))
    print('\t           Skew: [ {:.7f} ]'.format(K_opt[0,1]))
    print('\t     Distortion: [ {:.6f}  {:.6f} ]'.format(k_opt[0], k_opt[1]))

    util.info("Projection Matrices for WebGL:\n")
    znear, zfar = .1, 1000.
    for idx, e in enumerate(extrinsics_opt):
        p = util.get_camera_matrix(K_opt, e)
        util.info("P matrix for image " + str(idx + 1) + ":\n" + str(p))
        decomposed_p = util.decompose(p)
        webgl_p = util.to_opengl_projection(decomposed_p['intrinsic'], 0, 0, img_shapes[idx][0], img_shapes[idx][1], znear, zfar, direction="y down")
        util.info("P matrix(WebGl) for image " + str(idx + 1) + ":\n" + str(webgl_p))

    visualize.visualize_camera_frame(obj_points[0], extrinsics_opt[0], img_names[0])

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main()
