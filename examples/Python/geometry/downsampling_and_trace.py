# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/downsampling_and_trace.py

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    pcd = cv3d.io.read_point_cloud("../../TestData/fragment.ply")
    min_cube_size = 0.05
    print("\nOriginal, # of points %d" % (np.asarray(pcd.get_points()).shape[0]))
    pcd_down = pcd.voxel_down_sample(min_cube_size)
    print("\nScale %f, # of points %d" % \
          (min_cube_size, np.asarray(pcd_down.get_points()).shape[0]))
    min_bound = pcd_down.get_min_bound() - min_cube_size * 0.5
    max_bound = pcd_down.get_max_bound() + min_cube_size * 0.5

    pcd_curr = pcd_down
    num_scales = 3
    for i in range(1, num_scales):
        multiplier = pow(2, i)
        pcd_curr_down, cubic_id, original_indices = \
            pcd_curr.voxel_down_sample_and_trace(
                multiplier * min_cube_size, min_bound, max_bound, False)
        print("\nScale %f, # of points %d" %
              (multiplier * min_cube_size, np.asarray(pcd_curr_down.get_points()).shape[0]))
        print("Downsampled points (the first 10 points)")
        print(np.asarray(pcd_curr_down.get_points())[:10, :])
        print("Index (the first 10 indices)")
        print(np.asarray(cubic_id)[:10, :])

        print("Restore indices (the first 10 map indices)")
        map_indices = np.asarray([np.array(indices) for indices in original_indices])
        print(map_indices[:10])
        indices_final = np.concatenate(map_indices, axis=0)
        assert indices_final.shape[0] == pcd_curr.size()

        pcd_curr = pcd_curr_down
