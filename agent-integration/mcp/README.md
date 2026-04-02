# ACloudViewer MCP Server

Model Context Protocol (MCP) server for ACloudViewer. All headless operations
run via the **ACloudViewer binary** (`-SILENT` CLI mode), not any Python 3D library.

## Installation

```bash
pip install 'cli-anything-acloudviewer'
```

## Running

```bash
# Auto-detect mode (tries GUI RPC first, falls back to binary CLI)
cli-anything-acloudviewer-mcp

# Force headless (calls ACloudViewer binary, no GUI needed)
cli-anything-acloudviewer-mcp --mode headless

# Force GUI (requires running ACloudViewer with JSON-RPC plugin)
cli-anything-acloudviewer-mcp --mode gui --rpc-url ws://localhost:6001
```

## Agent Framework Configuration

### Cursor IDE (`.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "args": ["--mode", "auto"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add cli-anything-acloudviewer -- cli-anything-acloudviewer-mcp
```

### OpenClaw

```json
{
  "plugins": {
    "acloudviewer": {
      "command": "cli-anything-acloudviewer-mcp",
      "type": "mcp"
    }
  }
}
```

## Available Tools (178)

The MCP server registers **178 tools** covering headless CLI and GUI RPC operations. Below are the major categories — see `mcp_server.py` `list_tools()` for the full authoritative list.

### File I/O (5)
- **`open_file`**, **`convert_format`**, **`batch_convert`**, **`list_formats`**, **`export_entity`**

### Processing (10)
- **`subsample`**, **`compute_normals`**, **`sor_filter`**, **`crop`**, **`density`**, **`curvature`**, **`roughness`**, **`color_banding`**, **`stat_test`**, **`rasterize`**

### Scalar Fields (18)
- **`set_active_sf`**, **`remove_all_sfs`**, **`remove_sf`**, **`rename_sf`**, **`sf_arithmetic`**, **`sf_operation`**, **`coord_to_sf`**, **`sf_gradient`**, **`filter_sf`**, **`sf_color_scale`**, **`sf_convert_to_rgb`**, **`cloud_get_scalar_fields`**, **`cloud_set_active_sf`**, **`cloud_remove_sf`**, **`cloud_remove_all_sfs`**, **`cloud_rename_sf`**, **`cloud_filter_sf`**, **`cloud_coord_to_sf`**

### Normals (8)
- **`octree_normals`**, **`orient_normals_mst`**, **`invert_normals`**, **`clear_normals`**, **`normals_to_dip`**, **`normals_to_sfs`**, **`cloud_remove_normals_gui`**, **`cloud_invert_normals_gui`**

### Distance & Registration (3)
- **`c2c_distance`**, **`c2m_distance`**, **`icp_registration`**

### Geometry (6)
- **`extract_connected_components`**, **`approx_density`**, **`geometric_feature`**, **`moment`**, **`best_fit_plane`**, **`closest_point_set`**

### Mesh (14)
- **`delaunay`**, **`sample_mesh`**, **`mesh_volume`**, **`extract_vertices`**, **`flip_triangles`**, **`mesh_simplify`**, **`mesh_smooth`**, **`mesh_subdivide`**, **`mesh_sample_points`**, **`mesh_extract_vertices_gui`**, **`mesh_flip_triangles_gui`**, **`mesh_volume_gui`**, **`merge_meshes`**, **`mesh_merge_gui`**

### Merge & Cleanup (7)
- **`merge_clouds`**, **`cloud_merge_gui`**, **`remove_rgb`**, **`cloud_remove_rgb`**, **`remove_scan_grids`**, **`match_centers`**, **`drop_global_shift`**

### Plugin Processing (24)
- **`pcv`**, **`compass_export`**, **`compass_import_fol`**, **`compass_import_lin`**, **`compass_p21`**, **`compass_refit`**, **`sra`**, **`csf`**, **`ransac`**, **`m3c2`**, **`canupo`**, **`facets`**, **`hough_normals`**, **`poisson_recon`**, **`cork_boolean`**, **`voxfall`**, **`classify_3dmasc`**, **`treeiso`**, **`cloud_layers`**, **`animation`**, **`mplane`**, **`auto_seg`**, **`manual_seg`**, **`python_script`**

### IO Settings (8)
- **`draco_settings`**, **`e57_settings`**, **`las_settings`**, **`csv_matrix_settings`**, **`photoscan_settings`**, **`mesh_io_settings`**, **`core_io_settings`**, **`fbx_settings`**

### Colorimetric Segmentation (3)
- **`color_seg_rgb`**, **`color_seg_hsv`**, **`color_seg_scalar`**

### PCL Processing (18)
- **`pcl_sor`**, **`pcl_normal_estimation`**, **`pcl_mls`**, **`pcl_euclidean_cluster`**, **`pcl_sac_segmentation`**, **`pcl_region_growing`**, **`pcl_marching_cubes`**, **`pcl_greedy_triangulation`**, **`pcl_poisson_recon`**, **`pcl_convex_hull`**, **`pcl_don_segmentation`**, **`pcl_mincut_segmentation`**, **`pcl_fast_global_registration`**, **`pcl_extract_sift`**, **`pcl_projection_filter`**, **`pcl_general_filters`**, **`pcl_template_alignment`**, **`pcl_correspondence_matching`**

### Misc (4)
- **`g3point`**, **`volume_25d`**, **`crop_2d`**, **`bundler_import`**

### Colmap Reconstruction (13)
- **`colmap_auto_reconstruct`**, **`colmap_extract_features`**, **`colmap_match_features`**, **`colmap_sparse_reconstruct`**, **`colmap_undistort`**, **`colmap_dense_stereo`**, **`colmap_stereo_fusion`**, **`colmap_poisson_mesh`**, **`colmap_delaunay_mesh`**, **`colmap_image_texturer`**, **`colmap_model_converter`**, **`colmap_analyze_model`**, **`colmap_run`**

### SIBR Tools (12)
- **`sibr_viewer`**, **`sibr_tool`**, **`sibr_prepare_colmap`**, **`sibr_texture_mesh`**, **`sibr_unwrap_mesh`**, **`sibr_tonemapper`**, **`sibr_align_meshes`**, **`sibr_camera_converter`**, **`sibr_nvm_to_sibr`**, **`sibr_crop_from_center`**, **`sibr_clipping_planes`**, **`sibr_distord_crop`**

### Scene GUI (6)
- **`scene_list`**, **`scene_info`**, **`scene_remove`**, **`scene_set_visible`**, **`scene_select`**, **`scene_clear`**

### Entity GUI (2) · View GUI (7) · Cloud Painting (3) · Transform (2) · Utility (2)
- **`entity_rename`**, **`entity_set_color`**, **`screenshot`**, **`get_camera`**, **`view_set_orientation`**, **`view_zoom_fit`**, **`view_refresh`**, **`view_set_perspective`**, **`view_set_point_size`**, **`cloud_paint_uniform`**, **`cloud_paint_by_height`**, **`cloud_paint_by_scalar_field`**, **`transform_apply`**, **`transform_apply_file`**, **`get_info`**, **`list_rpc_methods`**

## Architecture

```
  AI Agent (Cursor / OpenClaw / Claude Code)
        ↓ MCP (stdio)
  cli-anything-acloudviewer-mcp
        ↓
  ACloudViewerBackend
        ↓           ↓
  GUI (RPC)     Headless (binary CLI)
  WebSocket     ACloudViewer -SILENT -O ... -SS ... -SAVE_CLOUDS
```
