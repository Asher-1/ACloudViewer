# CloudViewer: Asher-1.github.io

import numpy as np
import cloudViewer as cv3d


def auto_reconstruction(workspace_path, image_path, mask_path='', vocab_tree_path='', data_type='individual',
                        quality='high', mesher='poisson', camera_model='SIMPLE_RADIAL', single_camera=False,
                        sparse=True, dense=True, num_threads=-1, use_gpu=True,
                        gpu_index='-1'):
    """
    auto_reconstruction(workspace_path, image_path, mask_path='', vocab_tree_path='', data_type='individual',
    quality='high', mesher='poisson', camera_model='SIMPLE_RADIAL', single_camera=False,
    sparse=True, dense=True, num_threads=-1, use_gpu=True, gpu_index='-1')
    Function for the automatic reconstruction

    Args:
        workspace_path (str): The path to the workspace folder in which all results are stored.
        image_path (str): The path to the image folder which are used as input.
        mask_path (str, optional, default=''): The path to the mask folder which are used as input.
        vocab_tree_path (str, optional, default=''): The path to the vocabulary tree for feature matching.
        data_type (str, optional, default='individual'): Supported data types are {individual, video, internet}.
        quality (str, optional, default='high'): Supported quality types are {low, medium, high, extreme}.
        mesher (str, optional, default='poisson'): Supported meshing algorithm types are {poisson, delaunay}.
        camera_model (str, optional, default='SIMPLE_RADIAL'): Which camera model to use for images.
        single_camera (bool, optional, default=False): Whether to use shared intrinsics or not.
        sparse (bool, optional, default=True): Whether to perform sparse mapping.
        dense (bool, optional, default=True): Whether to perform dense mapping.
        num_threads (int, optional, default=-1): The number of threads to use in all stages.
        use_gpu (bool, optional, default=True): Whether to use the GPU in feature extraction and matching.
        gpu_index (str, optional, default='-1'): Index of the GPU used for GPU stages. For multi-GPU computation,
        you should separate multiple GPU indices by comma, e.g., ``0,1,2,3``. By default, all GPUs will be used in all stages.

    Returns:
        int
    """
    return cv3d.reconstruction.sfm.auto_reconstruction(workspace_path=workspace_path,
                                                       image_path=image_path,
                                                       mask_path=mask_path,
                                                       vocab_tree_path=vocab_tree_path,
                                                       data_type=data_type,
                                                       quality=quality, mesher=mesher,
                                                       camera_model=camera_model,
                                                       single_camera=single_camera,
                                                       sparse=sparse, dense=dense,
                                                       num_threads=num_threads,
                                                       use_gpu=use_gpu,
                                                       gpu_index=gpu_index)


def bundle_adjustment(input_path, output_path):
    """
    bundle_adjustment(input_path, output_path, bundle_adjustment_options=<cloudViewer.cuda.pybind.
    reconstruction.options.BundleAdjustmentOptions object at 0x7f82ad55ec38>)
    Function for the bundle adjustment

    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        bundle_adjustment_options (cloudViewer.reconstruction.options.BundleAdjustmentOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.BundleAdjustmentOptions object at 0x7f82ad55ec38>)

    Returns:
        int
    """
    bundle_adjustment_options = cv3d.reconstruction.options.BundleAdjustmentOptions()
    return cv3d.reconstruction.sfm.bundle_adjustment(input_path=input_path, output_path=output_path,
                                                     bundle_adjustment_options=bundle_adjustment_options)


def extract_color(image_path, input_path, output_path):
    """
    extract_color(image_path, input_path, output_path)
    Function for the extraction of images color

    Args:
        image_path (str): The path to the image folder which are used as input.
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.

    Returns:
        int
    """
    return cv3d.reconstruction.sfm.extract_color(image_path=image_path, input_path=input_path, output_path=output_path)


def filter_points(input_path, output_path, min_track_len=2, max_reproj_error=4.0, min_tri_angle=1.5):
    """
    filter_points(input_path, output_path, min_track_len=2, max_reproj_error=4.0, min_tri_angle=1.5)
    Function for the filtering of points

    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        min_track_len (int, optional, default=2): The minimum track length.
        max_reproj_error (float, optional, default=4.0): The maximum re-projection error.
        min_tri_angle (float, optional, default=1.5): The minimum tri angle.

    Returns:
        int
    """
    return cv3d.reconstruction.sfm.filter_points(input_path=input_path, output_path=output_path,
                                                 min_track_len=min_track_len, max_reproj_error=max_reproj_error,
                                                 min_tri_angle=min_tri_angle)


def hierarchical_mapper(database_path, image_path, output_path, num_workers=-1, image_overlap=50,
                        leaf_max_num_images=500):
    """
    hierarchical_mapper(database_path, image_path, output_path, num_workers=-1, image_overlap=50,
    leaf_max_num_images=500, incremental_mapper_options=<cloudViewer.cuda.pybind.
    reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55eca8>)
    Function for the hierarchical mapper

    Args:
        database_path (str): Path to database in which to store the extracted data
        image_path (str): The path to the image folder which are used as input.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        num_workers (int, optional, default=-1): The number of workers used to reconstruct clusters in parallel.
        image_overlap (int, optional, default=50): The number of overlapping images between child clusters.
        leaf_max_num_images (int, optional, default=500): The maximum number of images in a leaf node cluster,
         otherwise the cluster is further partitioned using the given branching factor.
         Note that a cluster leaf node will have at most `leaf_max_num_images + overlap`
          images to satisfy the overlap constraint.
        incremental_mapper_options (cloudViewer.reconstruction.options.IncrementalMapperOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55eca8>)

    Returns:
        int
    """
    incremental_mapper_options = cv3d.reconstruction.options.IncrementalMapperOptions()
    return cv3d.reconstruction.sfm.hierarchical_mapper(database_path=database_path, image_path=image_path,
                                                       output_path=output_path, num_workers=num_workers,
                                                       image_overlap=image_overlap,
                                                       leaf_max_num_images=leaf_max_num_images,
                                                       incremental_mapper_options=incremental_mapper_options)


def normal_mapper(database_path, image_path, input_path, output_path, image_list_path=''):
    """
    normal_mapper(database_path, image_path, input_path, output_path, image_list_path='',
    incremental_mapper_options=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55ec70>)
    Function for the normal mapper

    Args:
        database_path (str): Path to database in which to store the extracted data
        image_path (str): The path to the image folder which are used as input.
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        image_list_path (str, optional, default=''): A text file path containing image file path.
        incremental_mapper_options (cloudViewer.reconstruction.options.IncrementalMapperOptions, optional,
         default=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55ec70>)

    Returns:
        int
    """
    incremental_mapper_options = cv3d.reconstruction.options.IncrementalMapperOptions()
    return cv3d.reconstruction.sfm.normal_mapper(database_path=database_path, image_path=image_path,
                                                 input_path=input_path, output_path=output_path,
                                                 image_list_path=image_list_path,
                                                 incremental_mapper_options=incremental_mapper_options)


def rig_bundle_adjustment(input_path, output_path, rig_config_path, estimate_rig_relative_poses=True,
                          refine_relative_poses=True):
    """
    rig_bundle_adjustment(input_path, output_path, rig_config_path, estimate_rig_relative_poses=True,
    refine_relative_poses=True, bundle_adjustment_options=<cloudViewer.cuda.
    pybind.reconstruction.options.BundleAdjustmentOptions object at 0x7f82ad55ed18>)
    Function for the rig bundle adjustment

    Args:
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        rig_config_path (str): The rig config path.
        estimate_rig_relative_poses (bool, optional, default=True): Whether to estimate rig relative poses.
        refine_relative_poses (bool, optional, default=True): Whether to optimize the relative poses of the camera rigs.
        bundle_adjustment_options (cloudViewer.reconstruction.options.BundleAdjustmentOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.BundleAdjustmentOptions object at 0x7f82ad55ed18>)

    Returns:
        int
    """
    bundle_adjustment_options = cv3d.reconstruction.options.BundleAdjustmentOptions()
    return cv3d.reconstruction.sfm.rig_bundle_adjustment(input_path=input_path, output_path=output_path,
                                                         rig_config_path=rig_config_path,
                                                         estimate_rig_relative_poses=estimate_rig_relative_poses,
                                                         refine_relative_poses=refine_relative_poses,
                                                         bundle_adjustment_options=bundle_adjustment_options)


def triangulate_points(database_path, image_path, input_path, output_path, clear_points=False):
    """
    triangulate_points(database_path, image_path, input_path, output_path, clear_points=False,
    incremental_mapper_options=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55ece0>)
    Function for the triangulation of points

    Args:
        database_path (str): Path to database in which to store the extracted data
        image_path (str): The path to the image folder which are used as input.
        input_path (str): The input path containing cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        clear_points (bool, optional, default=False): Whether to clear all existing points and observations.
        incremental_mapper_options (cloudViewer.reconstruction.options.IncrementalMapperOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.IncrementalMapperOptions object at 0x7f82ad55ece0>)

    Returns:
        int
    """
    incremental_mapper_options = cv3d.reconstruction.options.IncrementalMapperOptions()
    return cv3d.reconstruction.sfm.triangulate_points(database_path=database_path, image_path=image_path,
                                                      input_path=input_path, output_path=output_path,
                                                      clear_points=clear_points,
                                                      incremental_mapper_options=incremental_mapper_options)


if __name__ == "__main__":
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    WORSPACE_PATH = "/media/asher/data/datasets/test"
    IMAGE_PATH = "/media/asher/data/datasets/dataset_monstree/mini6"
    VOCAB_TREE_PATH = "/media/asher/data/datasets/vocab_tree/vocab_tree_flickr100K_words32K.bin"

    flag = auto_reconstruction(WORSPACE_PATH, IMAGE_PATH, VOCAB_TREE_PATH)
    if flag != 0:
        print("clean_database failed!")
