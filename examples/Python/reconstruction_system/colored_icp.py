# Open3D: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/python/reconstruction_system/refine_registration.py

import numpy as np
import cloudViewer as cv3d
import sys

sys.path.append("../utility")
from visualization import draw_registration_result_original_color
import argparse


def multiscale_icp(source,
                   target,
                   voxel_size,
                   max_iter,
                   init_transformation=np.identity(4)):
    current_transformation = init_transformation
    for i, scale in enumerate(range(len(max_iter))):  # multi-scale approach
        iter = max_iter[scale]
        distance_threshold = 0.07
        print("voxel_size {}".format(voxel_size[scale]))
        source_down = source.voxel_down_sample(voxel_size[scale])
        target_down = target.voxel_down_sample(voxel_size[scale])
        source_down.estimate_normals(
            cv3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                  max_nn=30))
        target_down.estimate_normals(
            cv3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size[scale] * 2.0,
                                                  max_nn=30))
        result_icp = cv3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, distance_threshold,
            current_transformation,
            cv3d.pipelines.registration.TransformationEstimationForColoredICP(),
            cv3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))

        current_transformation = result_icp.transformation
        draw_registration_result_original_color(source, target,
                                                current_transformation)
        print(current_transformation)

    return result_icp.transformation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--voxel_size', default=0.05, type=float)
    args = parser.parse_args()

    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    source = cv3d.io.read_point_cloud(args.src)
    target = cv3d.io.read_point_cloud(args.dst)
    voxel_size = args.voxel_size

    trans = np.array([
        0.60949996, 0.49802465, -0.61683162, 0.15081973, -0.57239243,
        0.81477382, 0.09225254, -0.21454357, 0.5485223, 0.29684183, 0.78167015,
        -0.05018587, 0, 0, 0, 1
    ]).reshape((4, 4))

    trans = multiscale_icp(source, target,
                           [voxel_size, voxel_size / 2.0, voxel_size / 4.0],
                           [50, 30, 14], trans)