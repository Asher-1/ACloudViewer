// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "visualization/shader/GeometryRenderer.h"

#include <Image.h>
#include <ecvMesh.h>
#include <ecvBBox.h>
#include <ecvFacet.h>
#include <ecvHalfEdgeMesh.h>
#include <ecvPolyline.h>
#include <LineSet.h>
#include <ecvPointCloud.h>
#include <ecvOrientedBBox.h>
#include <ecvHObjectCaster.h>

#include "visualization/utility/PointCloudPicker.h"
#include "visualization/utility/SelectionPolygon.h"
#include "visualization/visualizer/RenderOptionWithEditing.h"

namespace cloudViewer {
namespace visualization {

namespace glsl {

bool PointCloudRenderer::Render(const RenderOption &option,
                                const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    const auto &pointcloud = (const ccPointCloud &)(*geometry_ptr_);
    bool success = true;
    if (pointcloud.hasNormals()) {
        if (option.point_color_option_ ==
            RenderOption::PointColorOption::Normal) {
            success &= normal_point_shader_.Render(pointcloud, option, view);
        } else {
            success &= phong_point_shader_.Render(pointcloud, option, view);
        }
        if (option.point_show_normal_) {
            success &=
                    simpleblack_normal_shader_.Render(pointcloud, option, view);
        }
    } else {
        success &= simple_point_shader_.Render(pointcloud, option, view);
    }
    return success;
}

bool PointCloudRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudRenderer::UpdateGeometry() {
    simple_point_shader_.InvalidateGeometry();
    phong_point_shader_.InvalidateGeometry();
    normal_point_shader_.InvalidateGeometry();
    simpleblack_normal_shader_.InvalidateGeometry();
    return true;
}

bool PointCloudPickingRenderer::Render(const RenderOption &option,
                                       const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    const auto &pointcloud = (const ccPointCloud &)(*geometry_ptr_);
    return picking_shader_.Render(pointcloud, option, view);
}

bool PointCloudPickingRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudPickingRenderer::UpdateGeometry() {
    picking_shader_.InvalidateGeometry();
    return true;
}

bool VoxelGridRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    if (option.mesh_show_wireframe_) {
        return simple_shader_for_voxel_grid_line_.Render(*geometry_ptr_, option,
                                                         view);
    } else {
        return simple_shader_for_voxel_grid_face_.Render(*geometry_ptr_, option,
                                                         view);
    }
}

bool VoxelGridRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::VOXEL_GRID)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool VoxelGridRenderer::UpdateGeometry() {
    simple_shader_for_voxel_grid_line_.InvalidateGeometry();
    simple_shader_for_voxel_grid_face_.InvalidateGeometry();
    return true;
}

bool OctreeRenderer::Render(const RenderOption &option,
                            const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    if (option.mesh_show_wireframe_) {
        return simple_shader_for_octree_line_.Render(*geometry_ptr_, option,
                                                     view);
    } else {
        bool rc = simple_shader_for_octree_face_.Render(*geometry_ptr_, option,
                                                        view);
        rc &= simple_shader_for_octree_line_.Render(*geometry_ptr_, option,
                                                    view);
        return rc;
    }
}

bool OctreeRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::POINT_OCTREE2)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool OctreeRenderer::UpdateGeometry() {
    simple_shader_for_octree_line_.InvalidateGeometry();
    simple_shader_for_octree_face_.InvalidateGeometry();
    return true;
}

bool LineSetRenderer::Render(const RenderOption &option,
                             const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    return simple_lineset_shader_.Render(*geometry_ptr_, option, view);
}

bool LineSetRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::LINESET)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool LineSetRenderer::UpdateGeometry() {
    simple_lineset_shader_.InvalidateGeometry();
    return true;
}

bool PolylineRenderer::Render(const RenderOption &option,
	const ViewControl &view) {
	if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
	return simple_polyline_shader_.Render(*geometry_ptr_, option, view);
}

bool PolylineRenderer::AddGeometry(
	std::shared_ptr<const ccHObject> geometry_ptr) {
	if (!geometry_ptr->isKindOf(CV_TYPES::POLY_LINE)) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool PolylineRenderer::UpdateGeometry() {
	simple_polyline_shader_.InvalidateGeometry();
	return true;
}

bool FacetRenderer::Render( const RenderOption &option,
							const ViewControl &view) {
	if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
	auto &facet = (ccFacet &)(*geometry_ptr_);

	// Normal Vector
	if (facet.normalVectorIsShown())
	{
		if(!simple_shader_for_normal_.Render(*facet.getNormalVectorMesh(), option, view)) return false;
	}

	// Contour
	if (!simple_polyline_shader_.Render(*facet.getContour(), option, view)) return false;

	// Polygon
	bool success = true;
	if (facet.getPolygon())
	{
		if (facet.getPolygon()->hasTriNormals() && facet.getPolygon()->hasNormals())
		{
			phong_shader_for_polygon_.Render(*facet.getPolygon(), option, view);
		}
		else
		{
			simple_shader_for_polygon_.Render(*facet.getPolygon(), option, view);
		}
	}

	return success;
}

bool FacetRenderer::AddGeometry(
	std::shared_ptr<const ccHObject> geometry_ptr) {
	if (!geometry_ptr->isKindOf(CV_TYPES::FACET)) {
		return false;
	}
	geometry_ptr_ = geometry_ptr;
	return UpdateGeometry();
}

bool FacetRenderer::UpdateGeometry() {
	simple_shader_for_normal_.InvalidateGeometry();
	phong_shader_for_polygon_.InvalidateGeometry();
	simple_shader_for_polygon_.InvalidateGeometry();
	simple_polyline_shader_.InvalidateGeometry();
	return true;
}

bool OrientedBoundingBoxRenderer::Render(const RenderOption &option,
                                         const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    return simple_oriented_bounding_box_shader_.Render(*geometry_ptr_, option, view);
}

bool OrientedBoundingBoxRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::ORIENTED_BBOX)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool OrientedBoundingBoxRenderer::UpdateGeometry() {
    simple_oriented_bounding_box_shader_.InvalidateGeometry();
    return true;
}

bool AxisAlignedBoundingBoxRenderer::Render(const RenderOption &option,
                                            const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    return simple_axis_aligned_bounding_box_shader_.Render(*geometry_ptr_,
                                                           option, view);
}

bool AxisAlignedBoundingBoxRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::BBOX)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool AxisAlignedBoundingBoxRenderer::UpdateGeometry() {
    simple_axis_aligned_bounding_box_shader_.InvalidateGeometry();
    return true;
}

bool TriangleMeshRenderer::Render(const RenderOption &option,
                                  const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    const auto &mesh = (const ccMesh &)(*geometry_ptr_);
    bool success = true;
    if (mesh.hasTriNormals() && mesh.hasNormals()) {
        if (option.mesh_color_option_ ==
            RenderOption::MeshColorOption::Normal) {
            success &= normal_mesh_shader_.Render(mesh, option, view);
        } else if (option.mesh_color_option_ ==
                           RenderOption::MeshColorOption::Color &&
                   mesh.hasTriangleUvs() && mesh.hasEigenTextures()) {
            success &= texture_phong_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= phong_mesh_shader_.Render(mesh, option, view);
        }
    } else {  // if normals are not ready
        if (option.mesh_color_option_ == RenderOption::MeshColorOption::Color &&
            mesh.hasTriangleUvs() && mesh.hasEigenTextures()) {
            success &= texture_simple_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= simple_mesh_shader_.Render(mesh, option, view);
        }
    }
    if (option.mesh_show_wireframe_) {
        success &= simpleblack_wireframe_shader_.Render(mesh, option, view);
    }
    return success;
}

bool TriangleMeshRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::MESH)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TriangleMeshRenderer::UpdateGeometry() {
    simple_mesh_shader_.InvalidateGeometry();
    texture_simple_mesh_shader_.InvalidateGeometry();
    phong_mesh_shader_.InvalidateGeometry();
    texture_phong_mesh_shader_.InvalidateGeometry();
    normal_mesh_shader_.InvalidateGeometry();
    simpleblack_wireframe_shader_.InvalidateGeometry();
    return true;
}


bool TetraMeshRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    return simple_tetramesh_shader_.Render(*geometry_ptr_, option, view);
}

bool TetraMeshRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::TETRA_MESH)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TetraMeshRenderer::UpdateGeometry() {
    simple_tetramesh_shader_.InvalidateGeometry();
    return true;
}

bool HalfEdgeMeshRenderer::Render(const RenderOption &option,
                                  const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    const auto &mesh = (const geometry::ecvHalfEdgeMesh &)(*geometry_ptr_);
    bool success = true;
    if (mesh.hasVertexNormals()) {
        if (option.mesh_color_option_ ==
            RenderOption::MeshColorOption::Normal) {
            success &= normal_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= phong_mesh_shader_.Render(mesh, option, view);
        }
    } else {  // if normals are not ready
        success &= simple_mesh_shader_.Render(mesh, option, view);
    }

    if (option.mesh_show_wireframe_) {
        success &= simpleblack_wireframe_shader_.Render(mesh, option, view);
    }
    return success;
}

bool HalfEdgeMeshRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool HalfEdgeMeshRenderer::UpdateGeometry() {
    simple_mesh_shader_.InvalidateGeometry();
    phong_mesh_shader_.InvalidateGeometry();
    normal_mesh_shader_.InvalidateGeometry();
    simpleblack_wireframe_shader_.InvalidateGeometry();
    return true;
}

bool ImageRenderer::Render(const RenderOption &option,
                           const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    return image_shader_.Render(*geometry_ptr_, option, view);
}

bool ImageRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::IMAGE2)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool ImageRenderer::UpdateGeometry() {
    image_shader_.InvalidateGeometry();
    return true;
}

bool RGBDImageRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    return rgbd_image_shader_.Render(*geometry_ptr_, option, view);
}

bool RGBDImageRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::RGBD_IMAGE)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool RGBDImageRenderer::UpdateGeometry() {
    rgbd_image_shader_.InvalidateGeometry();
    return true;
}

bool CoordinateFrameRenderer::Render(const RenderOption &option,
                                     const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    if (!option.show_coordinate_frame_) return true;
    return phong_shader_.Render(*geometry_ptr_, option, view);
}

bool CoordinateFrameRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::MESH) &&
        !geometry_ptr->isKindOf(CV_TYPES::HALF_EDGE_MESH)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool CoordinateFrameRenderer::UpdateGeometry() {
    phong_shader_.InvalidateGeometry();
    return true;
}

bool SelectionPolygonRenderer::Render(const RenderOption &option,
                                      const ViewControl &view) {
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    const auto &polygon = (const SelectionPolygon &)(*geometry_ptr_);
    if (polygon.isEmpty()) return true;
    if (!simple2d_shader_.Render(polygon, option, view)) return false;
    if (polygon.polygon_interior_mask_.isEmpty()) return true;
    return image_mask_shader_.Render(polygon.polygon_interior_mask_, option,
                                     view);
}

bool SelectionPolygonRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::CUSTOM_H_OBJECT)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool SelectionPolygonRenderer::UpdateGeometry() {
    simple2d_shader_.InvalidateGeometry();
    image_mask_shader_.InvalidateGeometry();
    return true;
}

bool PointCloudPickerRenderer::Render(const RenderOption &option,
                                      const ViewControl &view) {
    const int NUM_OF_COLOR_PALETTE = 5;
    Eigen::Vector3d color_palette[NUM_OF_COLOR_PALETTE] = {
            Eigen::Vector3d(255, 180, 0) / 255.0,
            Eigen::Vector3d(0, 166, 237) / 255.0,
            Eigen::Vector3d(246, 81, 29) / 255.0,
            Eigen::Vector3d(127, 184, 0) / 255.0,
            Eigen::Vector3d(13, 44, 84) / 255.0,
    };
    if (!is_visible_ || geometry_ptr_->isEmpty()) return true;
    const auto &picker = (const PointCloudPicker &)(*geometry_ptr_);
    const auto &pointcloud =
            (const ccPointCloud &)(*picker.pointcloud_ptr_);
    const auto &_option = (const RenderOptionWithEditing &)option;
    for (size_t i = 0; i < picker.picked_indices_.size(); i++) {
        size_t index = picker.picked_indices_[i];
        if (index < pointcloud.size()) {
            auto sphere = ccMesh::CreateSphere(
                    view.GetBoundingBox().getMaxExtent() *
                    _option.pointcloud_picker_sphere_size_);
			ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(sphere->getAssociatedCloud());
			assert(cloud);
			cloud->paintUniformColor(color_palette[i % NUM_OF_COLOR_PALETTE]);
            Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
            trans.block<3, 1>(0, 3) = pointcloud.getEigenPoint(index);
            sphere->transform(trans);
			sphere->computeVertexNormals();
            phong_shader_.InvalidateGeometry();
            if (!phong_shader_.Render(*sphere, option, view)) {
                return false;
            }
        }
    }
    return true;
}

bool PointCloudPickerRenderer::AddGeometry(
        std::shared_ptr<const ccHObject> geometry_ptr) {
    if (!geometry_ptr->isKindOf(CV_TYPES::CUSTOM_H_OBJECT)) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudPickerRenderer::UpdateGeometry() {
    // The geometry is updated on-the-fly
    // It is always in an invalidated status
    return true;
}

}  // namespace glsl

}  // namespace visualization
}  // namespace cloudViewer
