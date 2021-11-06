# encoding: utf-8

import numpy as np
import cloudViewer as cv3d


def align_model(input_path, output_path, database_path='', ref_images_path='', transform_path='',
                alignment_type='plane', max_error=0.0, min_common_images=3, robust_alignment=True,
                estimate_scale=True):
    """
    align_model(input_path, output_path, database_path='', ref_images_path='', transform_path='',
    alignment_type='plane', max_error=0.0, min_common_images=3, robust_alignment=True, estimate_scale=True)
    Function for the alignment of model
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        database_path (str, optional, default=''): Path to database in which to store the extracted data.
        ref_images_path (str, optional, default=''): Path to text file containing reference images per line.
        transform_path (str, optional, default=''): The alignment transformation matrix saving path.
        alignment_type (str, optional, default='plane'): Alignment type: supported values are {plane,
        ecef, enu, enu-unscaled, custom}.
        max_error (float, optional, default=0.0): Maximum error for a sample to be considered as an inlier.
        Note that the residual of an estimator corresponds to a squared error.
        min_common_images (int, optional, default=3): Minimum common images.
        robust_alignment (bool, optional, default=True): Whether align robustly or not.
        estimate_scale (bool, optional, default=True): Whether estimate scale or not.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.align_model(input_path=input_path, output_path=output_path,
                                                 database_path=database_path,
                                                 ref_images_path=ref_images_path,
                                                 transform_path=transform_path,
                                                 alignment_type=alignment_type,
                                                 max_error=max_error,
                                                 min_common_images=min_common_images,
                                                 robust_alignment=robust_alignment,
                                                 estimate_scale=estimate_scale)


def align_model_orientation(image_path, input_path, output_path, method='MANHATTAN-WORLD', max_image_size=1024):
    """
    align_model_orientation(image_path, input_path, output_path, method='MANHATTAN-WORLD', max_image_size=1024)
    Function for the orientation alignment of model
    
    Args:
        image_path (str)
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        method (str, optional, default='MANHATTAN-WORLD'):
        The supported Model Orientation Alignment values are {MANHATTAN-WORLD, IMAGE-ORIENTATION}.
        max_image_size (int, optional, default=1024): The maximum image size for line detection.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.align_model_orientation(image_path=image_path, input_path=input_path,
                                                             output_path=output_path, method=method,
                                                             max_image_size=max_image_size)


def analyze_model(input_path):
    """
    analyze_model(input_path)
    Function for the analyse of model
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.analyze_model(input_path=input_path)


def compare_model(input_path1, input_path2, output_path='', min_inlier_observations=0.3, max_reproj_error=8.0):
    """
    compare_model(input_path1, input_path2, output_path='', min_inlier_observations=0.3, max_reproj_error=8.0)
    Function for the comparison of model
    
    Args:
        input_path1 (str)
        input_path2 (str)
        output_path (str, optional, default=''): The output path containing target cameras.bin/txt,
        images.bin/txt and points3D.bin/txt.
        min_inlier_observations (float, optional, default=0.3): The threshold determines how many
        observations in a common image must reproject within the given threshold..
        max_reproj_error (float, optional, default=8.0): The Maximum re-projection error.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.compare_model(input_path1=input_path1, input_path2=input_path2,
                                                   output_path=output_path,
                                                   min_inlier_observations=min_inlier_observations,
                                                   max_reproj_error=max_reproj_error)


def convert_model(input_path, output_path, output_type, skip_distortion=False):
    """
    convert_model(input_path, output_path, output_type, skip_distortion=False)
    Function for the convertion of model
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_type (str): The supported output type values are {BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}.
        skip_distortion (bool, optional, default=False): Whether skip distortion or no.
        When skip_distortion == true it supports all camera models with the caveat that it's using the mean focal
        length which will be inaccurate for camera models with two focal lengths and distortion.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.convert_model(input_path=input_path, output_path=output_path,
                                                   output_type=output_type,
                                                   skip_distortion=skip_distortion)


def crop_model(input_path, output_path, boundary, gps_transform_path=''):
    """
    crop_model(input_path, output_path, boundary, gps_transform_path='')
    Function for the cropping of model
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        boundary (str): The cropping boundary coordinates.
        gps_transform_path (str, optional, default=''): The gps transformation parameters file path.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.crop_model(input_path=input_path, output_path=output_path, boundary=boundary,
                                                gps_transform_path=gps_transform_path)


def merge_model(input_path1, input_path2, output_path, max_reproj_error=64.0):
    """
    merge_model(input_path1, input_path2, output_path, max_reproj_error=64.0)
    Function for the merging of model
    
    Args:
        input_path1 (str)
        input_path2 (str)
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        max_reproj_error (float, optional, default=64.0): The Maximum re-projection error.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.merge_model(input_path1=input_path1, input_path2=input_path2,
                                                 output_path=output_path,
                                                 max_reproj_error=max_reproj_error)


def split_model(input_path, output_path, split_type, split_params, gps_transform_path='', min_reg_images=10,
                min_num_points=100, overlap_ratio=0.0, min_area_ratio=0.0, num_threads=-1):
    """
    split_model(input_path, output_path, split_type, split_params, gps_transform_path='', min_reg_images=10,
    min_num_points=100, overlap_ratio=0.0, min_area_ratio=0.0, num_threads=-1)
    Function for the splitting of model
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        split_type (str): The supported split type values are {tiles, extent, parts}.
        split_params (str): The split parameters file path.
        gps_transform_path (str, optional, default=''): The gps transformation parameters file path.
        min_reg_images (int, optional, default=10): The minimum number of reg images.
        min_num_points (int, optional, default=100): The minimum number of points.
        overlap_ratio (float, optional, default=0.0): The overlapped ratio.
        min_area_ratio (float, optional, default=0.0): The minimum area ratio.
        num_threads (int, optional, default=-1): The number of cpu thread.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.split_model(input_path=input_path, output_path=output_path, split_type=split_type,
                                                 split_params=split_params, gps_transform_path=gps_transform_path,
                                                 min_reg_images=min_reg_images, min_num_points=min_num_points,
                                                 overlap_ratio=overlap_ratio, min_area_ratio=min_area_ratio,
                                                 num_threads=num_threads)


def transform_model(input_path, output_path, transform_path, is_inverse=False):
    """
    transform_model(input_path, output_path, transform_path, is_inverse=False)
    Function for the transformation of model
    
    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        transform_path (str): The alignment transformation matrix saving path.
        is_inverse (bool, optional, default=False): Whether inverse or not.
    
    Returns:
        int
    """
    return cv3d.reconstruction.model.transform_model(input_path=input_path, output_path=output_path,
                                                     transform_path=transform_path, is_inverse=is_inverse)


if __name__ == '__main__':
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DATABASE_PATH = "/media/asher/data/datasets/gui_test/sql.db"
    INPUT_PATH = "/media/asher/data/datasets/gui_test/sparse/0"
    OUTPUT_PATH = "/media/asher/data/datasets/gui_test/model_out"

    flag = analyze_model(input_path=INPUT_PATH)
    if flag != 0:
        print("analyze_model failed!")

    flag = align_model(input_path=INPUT_PATH, output_path=OUTPUT_PATH,
                       database_path=DATABASE_PATH, max_error=0.8, alignment_type='plane')
    if flag != 0:
        print("align_model failed!")
