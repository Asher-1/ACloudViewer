// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// clang-format off
#include "visualization/utility/GLHelper.h"  // must include first!
#include "visualization/utility/SelectionPolygon.h"
// clang-format on

#include <Logging.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include "visualization/utility/SelectionPolygonVolume.h"
#include "visualization/visualizer/ViewControl.h"
#include "visualization/visualizer/ViewControlWithEditing.h"

namespace cloudViewer {
namespace visualization {

SelectionPolygon &SelectionPolygon::Clear() {
    polygon_.clear();
    is_closed_ = false;
    polygon_interior_mask_.Clear();
    polygon_type_ = SectionPolygonType::Unfilled;
    return *this;
}

bool SelectionPolygon::isEmpty() const {
    // A valid polygon, either close or open, should have at least 2 vertices.
    return polygon_.size() <= 1;
}

Eigen::Vector2d SelectionPolygon::GetMin2DBound() const {
    if (polygon_.empty()) {
        return Eigen::Vector2d(0.0, 0.0);
    }
    auto itr_x = std::min_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(0) < b(0);
            });
    auto itr_y = std::min_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(1) < b(1);
            });
    return Eigen::Vector2d((*itr_x)(0), (*itr_y)(1));
}

Eigen::Vector2d SelectionPolygon::GetMax2DBound() const {
    if (polygon_.empty()) {
        return Eigen::Vector2d(0.0, 0.0);
    }
    auto itr_x = std::max_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(0) < b(0);
            });
    auto itr_y = std::max_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(1) < b(1);
            });
    return Eigen::Vector2d((*itr_x)(0), (*itr_y)(1));
}

void SelectionPolygon::FillPolygon(int width, int height) {
    // Standard scan conversion code. See reference:
    // http://alienryderflex.com/polygon_fill/
    if (isEmpty()) return;
    is_closed_ = true;
    polygon_interior_mask_.Prepare(width, height, 1, 1);
    std::fill(polygon_interior_mask_.data_.begin(),
              polygon_interior_mask_.data_.end(), 0);
    std::vector<int> nodes;
    for (int y = 0; y < height; y++) {
        nodes.clear();
        for (size_t i = 0; i < polygon_.size(); i++) {
            size_t j = (i + 1) % polygon_.size();
            if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
                (polygon_[j](1) < y && polygon_[i](1) >= y)) {
                nodes.push_back(
                        (int)(polygon_[i](0) +
                              (y - polygon_[i](1)) /
                                      (polygon_[j](1) - polygon_[i](1)) *
                                      (polygon_[j](0) - polygon_[i](0)) +
                              0.5));
            }
        }
        std::sort(nodes.begin(), nodes.end());
        for (size_t i = 0; i < nodes.size(); i += 2) {
            if (nodes[i] >= width) {
                break;
            }
            if (nodes[i + 1] > 0) {
                if (nodes[i] < 0) nodes[i] = 0;
                if (nodes[i + 1] > width) nodes[i + 1] = width;
                for (int x = nodes[i]; x < nodes[i + 1]; x++) {
                    polygon_interior_mask_.data_[x + y * width] = 1;
                }
            }
        }
    }
}

std::shared_ptr<ccPointCloud> SelectionPolygon::CropPointCloud(
        const ccPointCloud &input, const ViewControl &view) {
    if (isEmpty()) {
        return cloudViewer::make_shared<ccPointCloud>();
    }
    switch (polygon_type_) {
        case SectionPolygonType::Rectangle:
            return CropPointCloudInRectangle(input, view);
        case SectionPolygonType::Polygon:
            return CropPointCloudInPolygon(input, view);
        case SectionPolygonType::Unfilled:
        default:
            return std::shared_ptr<ccPointCloud>();
    }
}

std::shared_ptr<ccMesh> SelectionPolygon::CropTriangleMesh(
        const ccMesh &input, const ViewControl &view) {
    if (isEmpty()) {
        return cloudViewer::make_shared<ccMesh>(nullptr);
    }
    if (input.size() == 0 && input.getAssociatedCloud()) {
        utility::LogWarning(
                "ccMesh contains vertices, but no triangles; "
                "cropping will always yield an empty "
                "ccMesh.");
        return cloudViewer::make_shared<ccMesh>(nullptr);
    }
    switch (polygon_type_) {
        case SectionPolygonType::Rectangle:
            return CropTriangleMeshInRectangle(input, view);
        case SectionPolygonType::Polygon:
            return CropTriangleMeshInPolygon(input, view);
        case SectionPolygonType::Unfilled:
        default:
            return std::shared_ptr<ccMesh>();
    }
}

std::shared_ptr<SelectionPolygonVolume>
SelectionPolygon::CreateSelectionPolygonVolume(const ViewControl &view) {
    auto volume = cloudViewer::make_shared<SelectionPolygonVolume>();
    const auto &editing_view = (const ViewControlWithEditing &)view;
    if (!editing_view.IsLocked() ||
        editing_view.GetEditingMode() == ViewControlWithEditing::FreeMode) {
        return volume;
    }
    int idx = 0;
    switch (editing_view.GetEditingMode()) {
        case ViewControlWithEditing::OrthoNegativeX:
        case ViewControlWithEditing::OrthoPositiveX:
            volume->orthogonal_axis_ = "X";
            idx = 0;
            break;
        case ViewControlWithEditing::OrthoNegativeY:
        case ViewControlWithEditing::OrthoPositiveY:
            volume->orthogonal_axis_ = "Y";
            idx = 1;
            break;
        case ViewControlWithEditing::OrthoNegativeZ:
        case ViewControlWithEditing::OrthoPositiveZ:
            volume->orthogonal_axis_ = "Z";
            idx = 2;
            break;
        default:
            break;
    }
    for (const auto &point : polygon_) {
        auto point3d = gl_util::Unproject(
                Eigen::Vector3d(point(0), point(1), 1.0), view.GetMVPMatrix(),
                view.GetWindowWidth(), view.GetWindowHeight());
        point3d(idx) = 0.0;
        volume->bounding_polygon_.push_back(point3d);
    }
    const auto &boundingbox = view.GetBoundingBox();
    double axis_len =
            boundingbox.GetMaxBound()(idx) - boundingbox.GetMinBound()(idx);
    volume->axis_min_ = boundingbox.GetMinBound()(idx) - axis_len;
    volume->axis_max_ = boundingbox.GetMaxBound()(idx) + axis_len;
    return volume;
}

std::shared_ptr<ccPointCloud> SelectionPolygon::CropPointCloudInRectangle(
        const ccPointCloud &input, const ViewControl &view) {
    return input.SelectByIndex(CropInRectangle(input.getPoints(), view));
}

std::shared_ptr<ccPointCloud> SelectionPolygon::CropPointCloudInPolygon(
        const ccPointCloud &input, const ViewControl &view) {
    return input.SelectByIndex(CropInPolygon(input.getPoints(), view));
}

std::shared_ptr<ccMesh> SelectionPolygon::CropTriangleMeshInRectangle(
        const ccMesh &input, const ViewControl &view) {
    return input.SelectByIndex(CropInRectangle(input.getVertices(), view));
}

std::shared_ptr<ccMesh> SelectionPolygon::CropTriangleMeshInPolygon(
        const ccMesh &input, const ViewControl &view) {
    return input.SelectByIndex(CropInPolygon(input.getVertices(), view));
}

std::vector<size_t> SelectionPolygon::CropInRectangle(
        const std::vector<CCVector3> &input, const ViewControl &view) {
    std::vector<size_t> output_index;
    Eigen::Matrix4d mvp_matrix = view.GetMVPMatrix().cast<double>();
    double half_width = (double)view.GetWindowWidth() * 0.5;
    double half_height = (double)view.GetWindowHeight() * 0.5;
    auto min_bound = GetMin2DBound();
    auto max_bound = GetMax2DBound();
    utility::ConsoleProgressBar progress_bar((int64_t)input.size(),
                                             "Cropping geometry: ");
    for (size_t i = 0; i < input.size(); i++) {
        ++progress_bar;
        const auto &point = input[i];
        Eigen::Vector4d pos =
                mvp_matrix * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) == 0.0) break;
        pos /= pos(3);
        double x = (pos(0) + 1.0) * half_width;
        double y = (pos(1) + 1.0) * half_height;
        if (x >= min_bound(0) && x <= max_bound(0) && y >= min_bound(1) &&
            y <= max_bound(1)) {
            output_index.push_back(i);
        }
    }
    return output_index;
}

std::vector<size_t> SelectionPolygon::CropInPolygon(
        const std::vector<CCVector3> &input, const ViewControl &view) {
    std::vector<size_t> output_index;
    Eigen::Matrix4d mvp_matrix = view.GetMVPMatrix().cast<double>();
    double half_width = (double)view.GetWindowWidth() * 0.5;
    double half_height = (double)view.GetWindowHeight() * 0.5;
    std::vector<double> nodes;
    utility::ConsoleProgressBar progress_bar((int64_t)input.size(),
                                             "Cropping geometry: ");
    for (size_t k = 0; k < input.size(); k++) {
        ++progress_bar;
        const auto &point = input[k];
        Eigen::Vector4d pos =
                mvp_matrix * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) == 0.0) break;
        pos /= pos(3);
        double x = (pos(0) + 1.0) * half_width;
        double y = (pos(1) + 1.0) * half_height;
        nodes.clear();
        for (size_t i = 0; i < polygon_.size(); i++) {
            size_t j = (i + 1) % polygon_.size();
            if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
                (polygon_[j](1) < y && polygon_[i](1) >= y)) {
                nodes.push_back(polygon_[i](0) +
                                (y - polygon_[i](1)) /
                                        (polygon_[j](1) - polygon_[i](1)) *
                                        (polygon_[j](0) - polygon_[i](0)));
            }
        }
        std::sort(nodes.begin(), nodes.end());
        auto loc = std::lower_bound(nodes.begin(), nodes.end(), x);
        if (std::distance(nodes.begin(), loc) % 2 == 1) {
            output_index.push_back(k);
        }
    }
    return output_index;
}

}  // namespace visualization
}  // namespace cloudViewer
