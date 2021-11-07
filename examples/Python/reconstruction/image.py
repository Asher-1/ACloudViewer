# encoding: utf-8

import numpy as np
import cloudViewer as cv3d


def delete_image(input_path, output_path, image_ids_path='',
                 image_names_path=''):
    """
    delete_image(input_path, output_path, image_ids_path='', image_names_path='')
    Function for the deletion of images
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        image_ids_path (str, optional, default=''): Path to text file containing one image_id to delete per line.
        image_names_path (str, optional, default=''): Path to text file containing one image name to delete per line.
    
    Returns:
        int
    """
    return cv3d.reconstruction.image.delete_image(input_path=input_path, output_path=output_path,
                                                  image_ids_path=image_ids_path, image_names_path=image_names_path)


def filter_image(input_path, output_path, min_focal_length_ratio=0.1, max_focal_length_ratio=10.0,
                 max_extra_param=100.0, min_num_observations=10):
    """
    filter_image(input_path, output_path, min_focal_length_ratio=0.1,
    max_focal_length_ratio=10.0, max_extra_param=100.0, min_num_observations=10)
    Function for the filtering of images
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        min_focal_length_ratio (float, optional, default=0.1): Minimum ratio of focal length over minimum sensor dimension.
        max_focal_length_ratio (float, optional, default=10.0): Maximum ratio of focal length over maximum sensor dimension.
        max_extra_param (float, optional, default=100.0): Maximum magnitude of each extra parameter.
        min_num_observations (int, optional, default=10): The maximum number of observations.
    
    Returns:
        int
    """
    return filter_image(input_path=input_path, output_path=output_path,
                        min_focal_length_ratio=min_focal_length_ratio,
                        max_focal_length_ratio=max_focal_length_ratio,
                        max_extra_param=max_extra_param,
                        min_num_observations=min_num_observations)


def rectify_image(image_path, input_path, output_path, stereo_pairs_list,
                  blank_pixels=0.0, min_scale=0.2,
                  max_scale=2.0, max_image_size=-1):
    """
    rectify_image(image_path, input_path, output_path, stereo_pairs_list, blank_pixels=0.0,
                    min_scale=0.2, max_scale=2.0, max_image_size=-1)
    Function for the rectification of images
    
    Args:
        image_path (str)
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        stereo_pairs_list (str): A text file path containing stereo image pair names from.
        The text file is expected to have one image pair per line, e.g.:
        image_name1.jpg image_name2.jpg
        image_name3.jpg image_name4.jpg
        image_name5.jpg image_name6.jpg
        blank_pixels (float, optional, default=0.0): The amount of blank pixels in the undistorted image in the range [0, 1].
        min_scale (float, optional, default=0.2): Minimum scale change of camera used to satisfy the blank pixel constraint.
        max_scale (float, optional, default=2.0): Maximum scale change of camera used to satisfy the blank pixel constraint.
        max_image_size (int, optional, default=-1): Maximum image size in terms of width or height of the undistorted camera.
    
    Returns:
        int
    """
    return rectify_image(image_path=image_path, input_path=input_path, output_path=output_path,
                         stereo_pairs_list=stereo_pairs_list, blank_pixels=blank_pixels, min_scale=min_scale,
                         max_scale=max_scale, max_image_size=max_image_size)


def register_image(database_path, input_path, output_path):
    """
    register_image(database_path, input_path, output_path,
    incremental_mapper_options=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55e810>)
    Function for the registeration of images
    
    Args:
        database_path (str)
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        incremental_mapper_options (cloudViewer.reconstruction.options.IncrementalMapperOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55e810>)
    
    Returns:
        int
    """
    incremental_mapper_options = cv3d.reconstruction.options.IncrementalMapperOptions()
    return cv3d.reconstruction.image.register_image(database_path=database_path,
                                                    input_path=input_path,
                                                    output_path=output_path,
                                                    incremental_mapper_options=incremental_mapper_options)


def undistort_image(image_path, input_path, output_path, image_list_path='', output_type='COLMAP',
                    copy_policy='copy', num_patch_match_src_images=20, blank_pixels=0.0, min_scale=0.2,
                    max_scale=2.0, max_image_size=-1, roi_min_x=0.0, roi_min_y=0.0, roi_max_x=1.0, roi_max_y=1.0):
    """
    undistort_image(image_path, input_path, output_path, image_list_path='', output_type='COLMAP', copy_policy='copy',
     num_patch_match_src_images=20.0, blank_pixels=0.0, min_scale=0.2, max_scale=2.0,
     max_image_size=-1, roi_min_x=0.0, roi_min_y=0.0, roi_max_x=1.0, roi_max_y=1.0)
    Function for the undistortion of images
    
    Args:
        image_path (str)
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        image_list_path (str, optional, default=''): A text file path containing image file path.
        output_type (str, optional, default='COLMAP'): Output file format: supported values are {'COLMAP', 'PMVS', 'CMP-MVS'}.
        copy_policy (str, optional, default='copy'): Supported copy policy are {copy, soft-link, hard-link}.
        num_patch_match_src_images (int, optional, default=20.0): The number of patch match source images.
        blank_pixels (float, optional, default=0.0): The amount of blank pixels in the undistorted image in the range [0, 1].
        min_scale (float, optional, default=0.2): Minimum scale change of camera used to satisfy the blank pixel constraint.
        max_scale (float, optional, default=2.0): Maximum scale change of camera used to satisfy the blank pixel constraint.
        max_image_size (int, optional, default=-1): Maximum image size in terms of width or height of the undistorted camera.
        roi_min_x (float, optional, default=0.0): The value in the range [0, 1] that define the ROI (region of interest) minimum x in original image.
        roi_min_y (float, optional, default=0.0): The value in the range [0, 1] that define the ROI (region of interest) minimum y in original image.
        roi_max_x (float, optional, default=1.0): The value in the range [0, 1] that define the ROI (region of interest) maximum x in original image.
        roi_max_y (float, optional, default=1.0): The value in the range [0, 1] that define the ROI (region of interest) maximum y in original image.
    
    Returns:
        int
    """
    return cv3d.reconstruction.image.undistort_image(image_path=image_path,
                                                     input_path=input_path,
                                                     output_path=output_path,
                                                     image_list_path=image_list_path,
                                                     output_type=output_type,
                                                     copy_policy=copy_policy,
                                                     num_patch_match_src_images=num_patch_match_src_images,
                                                     blank_pixels=blank_pixels, min_scale=min_scale,
                                                     max_scale=max_scale, max_image_size=max_image_size,
                                                     roi_min_x=roi_min_x, roi_min_y=roi_min_y,
                                                     roi_max_x=roi_max_x, roi_max_y=roi_max_y)


def undistort_image_standalone(image_path, input_path, output_path, blank_pixels=0.0, min_scale=0.2, max_scale=2.0,
                               max_image_size=-1, roi_min_x=0.0, roi_min_y=0.0, roi_max_x=1.0, roi_max_y=1.0):
    """
    undistort_image_standalone(image_path, input_path, output_path, blank_pixels=0.0, min_scale=0.2, max_scale=2.0,
     max_image_size=-1, roi_min_x=0.0, roi_min_y=0.0, roi_max_x=1.0, roi_max_y=1.0)
    Function for the standalone undistortion of images
    
    Args:
        image_path (str)
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        blank_pixels (float, optional, default=0.0): The amount of blank pixels in the undistorted image in the range [0, 1].
        min_scale (float, optional, default=0.2): Minimum scale change of camera used to satisfy the blank pixel constraint.
        max_scale (float, optional, default=2.0): Maximum scale change of camera used to satisfy the blank pixel constraint.
        max_image_size (int, optional, default=-1): Maximum image size in terms of width or height of the undistorted camera.
        roi_min_x (float, optional, default=0.0): The value in the range [0, 1] that define the ROI (region of interest) minimum x in original image.
        roi_min_y (float, optional, default=0.0): The value in the range [0, 1] that define the ROI (region of interest) minimum y in original image.
        roi_max_x (float, optional, default=1.0): The value in the range [0, 1] that define the ROI (region of interest) maximum x in original image.
        roi_max_y (float, optional, default=1.0): The value in the range [0, 1] that define the ROI (region of interest) maximum y in original image.
    
    Returns:
        int
    """
    return cv3d.reconstruction.image.undistort_image_standalone(image_path=image_path, input_path=input_path,
                                                                output_path=output_path, blank_pixels=blank_pixels,
                                                                min_scale=min_scale, max_scale=max_scale,
                                                                max_image_size=max_image_size,
                                                                roi_min_x=roi_min_x, roi_min_y=roi_min_y,
                                                                roi_max_x=roi_max_x, roi_max_y=roi_max_y)


if __name__ == '__main__':
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DATABASE_PATH = "/media/asher/data/datasets/gui_test/database.db"
    IMAGE_PATH = "/media/asher/data/datasets/dataset_monstree/mini6"
    INPUT_PATH = "/media/asher/data/datasets/gui_test/sparse/0"

    INPUT_FILE = "/media/asher/data/datasets/gui_test/input_file.txt"
    OUTPUT_PATH = "/media/asher/data/datasets/gui_test/undistorted"
    STANDALONE_OUTPUT_PATH = "/media/asher/data/datasets/gui_test/undistorted/standalone"

    flag = undistort_image(image_path=IMAGE_PATH, input_path=INPUT_PATH,
                           output_path=OUTPUT_PATH, image_list_path="", output_type='COLMAP')
    if flag != 0:
        print("undistort_image failed!")

    flag = undistort_image_standalone(image_path=IMAGE_PATH, input_path=INPUT_FILE,
                                      output_path=STANDALONE_OUTPUT_PATH)
    if flag != 0:
        print("undistort_image_standalone failed!")
