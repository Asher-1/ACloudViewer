import cloudViewer as cv3d
import numpy as np
from os import listdir, makedirs
from os.path import exists, isfile, join, splitext, dirname, basename
import re
import struct
import os
import argparse


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


# get list of files inside a folder, matching the externsion, in sorted order.
def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


# converts kitti binary to pcd.
def bin_to_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = cv3d.geometry.ccPointCloud()
    pcd.set_points(cv3d.utility.Vector3dVector(np_pcd))
    return pcd


# preprocess and save in .ply format.
def preprocess_and_save(source_folder,
                        destination_folder,
                        voxel_size=0.05,
                        start_idx=0,
                        end_idx=1000):
    # get all files from the folder, and sort by name.
    filenames = get_file_list(source_folder, ".bin")

    print(
        "Converting .bin to .ply files and pre-processing from frame {} to index {}"
        .format(start_idx, end_idx))

    if end_idx < start_idx:
        raise RuntimeError("End index must be smaller than start index.")
    if end_idx > len(filenames):
        end_idx = len(filenames)
        print(
            "WARNING: End index is greater than total file length, taking file length as end index."
        )

    filenames = filenames[start_idx:end_idx]
    for path in filenames:
        # convert kitti bin format to pcd format.
        pcd = bin_to_pcd(path)

        # downsample and estimate normals.
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
        voxel_down_pcd.estimate_normals(
            search_param=cv3d.geometry.KDTreeSearchParamKNN(),
            fast_normal_computation=False)

        # convert to Float32 dtype.
        tpcd = cv3d.t.geometry.PointCloud.from_legacy(voxel_down_pcd)
        tpcd.point["points"] = tpcd.point["points"].to(cv3d.core.Dtype.Float32)
        tpcd.point["normals"] = tpcd.point["normals"].to(
            cv3d.core.Dtype.Float32)

        # extract name from path.
        name = str(path).rsplit('/', 1)[-1]
        name = name[:-3] + "ply"

        # write to the destination folder.
        output_path = destination_folder + name
        cv3d.t.io.write_point_cloud(output_path, tpcd)


if __name__ == '__main__':
    DATA_PATH = "/media/asher/data/datasets/3d_data/KITTI_odometry/data_odometry_velodyne/sequences/1-10/01"
    DESTINATION_PATH = "/media/asher/data/datasets/3d_data/KITTI_odometry/data_odometry_velodyne/pcd_sequences/1_10/01"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default=DATA_PATH,
        help='Kitti city sequence name [Example: "2011_09_26_drive_0009"].')
    parser.add_argument(
        '--destination_path',
        type=str,
        default=DESTINATION_PATH,
        help='Kitti city sequence name [Example: "2011_09_26_drive_0009"].')
    parser.add_argument('--voxel_size',
                        type=float,
                        default=0.05,
                        help='voxel size of the pointcloud.')
    parser.add_argument('--start_index',
                        type=int,
                        default=0,
                        help='start index of the dataset frame.')
    parser.add_argument('--end_index',
                        type=int,
                        default=1000,
                        help='maximum end index of the dataset frame.')

    args = parser.parse_args()

    source_folder = args.data_path
    destination_path = args.destination_path

    # get source path to raw dataset, and target path to processed dataset.
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    else:
        for f in os.listdir(destination_path):
            os.remove(os.path.join(destination_path, f))

    print("Source raw kitti lidar data: ", source_folder)

    # convert bin to pcd, pre-process and save.
    preprocess_and_save(source_folder, destination_path, args.voxel_size,
                        args.start_index, args.end_index)

    print("Data fetching completed. Output pointcloud frames: ",
          destination_path)
