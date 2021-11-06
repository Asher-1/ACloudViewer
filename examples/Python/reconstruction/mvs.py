# encoding: utf-8

import numpy as np
import cloudViewer as cv3d


def mesh_delaunay(input_path, output_path, input_type='dense'):
    """
    mesh_delaunay(input_path, output_path, input_type='dense', delaunay_meshing_options=<cloudViewer.cuda
    .pybind.reconstruction.options.DelaunayMeshingOptions object at 0x7f82ad55e960>)
    Function for the delaunay of mesh
    
    Args:
        input_path (str): Path to either the dense workspace folder or the sparse reconstruction.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        input_type (str, optional, default='dense'): Supported input type values are {dense, sparse}.
        delaunay_meshing_options (cloudViewer.reconstruction.options.DelaunayMeshingOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.DelaunayMeshingOptions object at 0x7f82ad55e960>)
    
    Returns:
        int
    """
    delaunay_meshing_options = cv3d.reconstruction.options.DelaunayMeshingOptions()
    return cv3d.reconstruction.mvs.mesh_delaunay(input_path=input_path,
                                                 output_path=output_path,
                                                 input_type=input_type,
                                                 delaunay_meshing_options=delaunay_meshing_options)


def poisson_mesh(input_path, output_path):
    """
    poisson_mesh(input_path, output_path, poisson_meshing_options=<cloudViewer.cuda.pybind.reconstruction.
    options.PoissonMeshingOptions object at 0x7f82ad55ea40>)
    Function for the poisson of mesh
    
    Args:
        input_path (str): Path to either the dense workspace folder or the sparse reconstruction.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        poisson_meshing_options (cloudViewer.reconstruction.options.PoissonMeshingOptions, optional,
         default=<cloudViewer.cuda.pybind.reconstruction.options.PoissonMeshingOptions object at 0x7f82ad55ea40>)
    
    Returns:
        int
    """
    poisson_meshing_options = cv3d.reconstruction.options.PoissonMeshingOptions()
    return cv3d.reconstruction.mvs.poisson_mesh(input_path=input_path, output_path=output_path,
                                                poisson_meshing_options=poisson_meshing_options)


def stereo_fuse(workspace_path, output_path, bbox_path='', stereo_input_type='geometric', output_type='PLY',
                workspace_format='COLMAP', pmvs_option_name='option-all'):
    """
    stereo_fuse(workspace_path, output_path, bbox_path='', stereo_input_type='geometric', output_type='PLY',
     workspace_format='COLMAP', pmvs_option_name='option-all',
     stereo_fusion_options=<cloudViewer.cuda.pybind.reconstruction.options.StereoFusionOptions object at 0x7f82ad55ea78>)
    Function for the stereo path-match of mesh
    
    Args:
        workspace_path (str): Path to the folder containing the undistorted images.
        output_path (str): The output path containing target cameras.bin/txt, images.bin/txt and points3D.bin/txt.
        bbox_path (str, optional, default=''): The bounds file path.
        stereo_input_type (str, optional, default='geometric'): Supported stereo input type values are {photometric, geometric}.
        output_type (str, optional, default='PLY'): Supported output type values are {BIN, TXT, PLY}.
        workspace_format (str, optional, default='COLMAP'): Supported workspace format values are {COLMAP, PMVS}.
        pmvs_option_name (str, optional, default='option-all'): The pmvs option name.
        stereo_fusion_options (cloudViewer.reconstruction.options.StereoFusionOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.StereoFusionOptions object at 0x7f82ad55ea78>)

    Returns:
        int
    """

    stereo_fusion_options = cv3d.reconstruction.options.StereoFusionOptions()
    return cv3d.reconstruction.mvs.stereo_fuse(workspace_path=workspace_path,
                                               output_path=output_path,
                                               bbox_path=bbox_path,
                                               stereo_input_type=stereo_input_type,
                                               output_type=output_type,
                                               workspace_format=workspace_format,
                                               pmvs_option_name=pmvs_option_name,
                                               stereo_fusion_options=stereo_fusion_options)


def stereo_patch_match(workspace_path, config_path='', workspace_format='COLMAP', pmvs_option_name='option-all'):
    """
    stereo_patch_match(workspace_path, config_path='', workspace_format='COLMAP', pmvs_option_name='option-all',
    patch_match_options=<cloudViewer.cuda.pybind.reconstruction.options.PatchMatchOptions object at 0x7f82ad55e9d0>)
    Function for the stereo path-match of mesh

    Args:
        workspace_path (str): Path to the folder containing the undistorted images.
        config_path (str, optional, default=''): The config path.
        workspace_format (str, optional, default='COLMAP'): Supported workspace format values are {COLMAP, PMVS}.
        pmvs_option_name (str, optional, default='option-all'): The pmvs option name.
        patch_match_options (cloudViewer.reconstruction.options.PatchMatchOptions, optional,
        default=<cloudViewer.cuda.pybind.reconstruction.options.PatchMatchOptions object at 0x7f82ad55e9d0>)

    Returns:
        int
    """
    patch_match_options = cv3d.reconstruction.options.PatchMatchOptions()
    return cv3d.reconstruction.mvs.stereo_patch_match(workspace_path=workspace_path,
                                                      config_path=config_path,
                                                      workspace_format=workspace_format,
                                                      pmvs_option_name=pmvs_option_name,
                                                      patch_match_options=patch_match_options)


if __name__ == '__main__':
    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    DELAUNAY_INPUT_PATH = "/media/asher/data/datasets/gui_test/dense/0"
    DELAUNAY_OUTPUT_PATH = "/media/asher/data/datasets/gui_test/meshes/delaunay_meshed.ply"

    POISSON_INPUT_PATH = "/media/asher/data/datasets/gui_test/dense/0/fused.ply"
    POISSON_OUTPUT_PATH = "/media/asher/data/datasets/gui_test/meshes/poisson_meshed.ply"

    flag = mesh_delaunay(input_path=DELAUNAY_INPUT_PATH, output_path=DELAUNAY_OUTPUT_PATH, input_type='dense')
    if flag != 0:
        print("mesh_delaunay failed!")
    else:
        print("mesh_delaunay successfully!")

    flag = poisson_mesh(input_path=POISSON_INPUT_PATH, output_path=POISSON_OUTPUT_PATH)
    if flag != 0:
        print("poisson_mesh failed!")
    else:
        print("poisson_mesh successfully!")
