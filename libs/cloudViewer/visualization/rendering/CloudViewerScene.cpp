// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "visualization/rendering/CloudViewerScene.h"

#include <LineSet.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <algorithm>

#include "visualization/gui/Application.h"
#include "visualization/rendering/MaterialRecord.h"
#include "visualization/rendering/Scene.h"
#include "visualization/rendering/View.h"

namespace cloudViewer {
namespace visualization {
namespace rendering {

const std::string kAxisObjectName("__axis__");
const std::string kFastModelObjectSuffix("__fast__");
const std::string kLowQualityModelObjectSuffix("__low__");

namespace {
std::shared_ptr<ccMesh> CreateAxisGeometry(double axis_length) {
    const double sphere_radius = 0.005 * axis_length;
    const double cyl_radius = 0.0025 * axis_length;
    const double cone_radius = 0.0075 * axis_length;
    const double cyl_height = 0.975 * axis_length;
    const double cone_height = 0.025 * axis_length;

    auto mesh_frame = ccMesh::CreateSphere(sphere_radius);
    mesh_frame->computeVertexNormals();
    mesh_frame->paintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));

    std::shared_ptr<ccMesh> mesh_arrow;
    Eigen::Matrix4d transformation;

    mesh_arrow = ccMesh::CreateArrow(cyl_radius, cone_radius, cyl_height,
                                     cone_height);
    mesh_arrow->computeVertexNormals();
    mesh_arrow->paintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
    transformation << 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    mesh_arrow->transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = ccMesh::CreateArrow(cyl_radius, cone_radius, cyl_height,
                                     cone_height);
    mesh_arrow->computeVertexNormals();
    mesh_arrow->paintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
    transformation << 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1;
    mesh_arrow->transform(transformation);
    *mesh_frame += *mesh_arrow;

    mesh_arrow = ccMesh::CreateArrow(cyl_radius, cone_radius, cyl_height,
                                     cone_height);
    mesh_arrow->computeVertexNormals();
    mesh_arrow->paintUniformColor(Eigen::Vector3d(0.0, 0.0, 1.0));
    transformation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mesh_arrow->transform(transformation);
    *mesh_frame += *mesh_arrow;

    // Add UVs because material shader for axes expects them
    mesh_frame->triangle_uvs_.resize(mesh_frame->size() * 3, {0.0, 0.0});

    return mesh_frame;
}

void RecreateAxis(Scene* scene, const ccBBox& bounds, bool enabled) {
    scene->RemoveGeometry(kAxisObjectName);

    // Axes length should be the longer of the bounds extent or 25% of the
    // distance from the origin. The latter is necessary so that the axis is
    // big enough to be visible even if the object is far from the origin.
    // See caterpillar.ply from Tanks & Temples.
    auto axis_length = bounds.getMaxExtent();
    if (axis_length < 0.001) {  // avoid div by zero errors in CreateAxes()
        axis_length = 1.0;
    }
    axis_length = std::max(axis_length, 0.25f * bounds.getCenter().norm());
    auto mesh = CreateAxisGeometry(axis_length);
    MaterialRecord mat;
    mat.shader = "defaultUnlit";
    scene->AddGeometry(kAxisObjectName, *mesh, mat);
    // It looks awkward to have the axis cast a a shadow, and even stranger
    // to receive a shadow.
    scene->GeometryShadows(kAxisObjectName, false, false);
    scene->ShowGeometry(kAxisObjectName, enabled);
}

}  // namespace

CloudViewerScene::CloudViewerScene(Renderer& renderer) : renderer_(renderer) {
    scene_ = renderer_.CreateScene();
    auto scene = renderer_.GetScene(scene_);
    view_ = scene->AddView(0, 0, 1, 1);
    SetBackground({1.0f, 1.0f, 1.0f, 1.0f});

    SetLighting(LightingProfile::MED_SHADOWS, {0.577f, -0.577f, -0.577f});

    RecreateAxis(scene, bounds_, false);
}

CloudViewerScene::~CloudViewerScene() {
    ClearGeometry();
    auto scene = renderer_.GetScene(scene_);
    scene->RemoveGeometry(kAxisObjectName);
    scene->RemoveView(view_);
}

View* CloudViewerScene::GetView() const {
    auto scene = renderer_.GetScene(scene_);
    return scene->GetView(view_);
}

void CloudViewerScene::SetViewport(std::int32_t x,
                                   std::int32_t y,
                                   std::uint32_t width,
                                   std::uint32_t height) {
    // Setup the view in which we render to a texture. Since this is just a
    // texture, we want our viewport to be the entire texture.
    auto view = GetView();
    // Since we are rendering into a texture (EnableViewCaching(true) below),
    // we need to use the entire texture; the viewport passed in is the viewport
    // with respect to the window, and we are setting the viewport with respect
    // to the render target here.
    view->SetViewport(0, 0, width, height);
    view->EnableViewCaching(true);
}

void CloudViewerScene::ShowSkybox(bool enable) {
    auto scene = renderer_.GetScene(scene_);
    scene->ShowSkybox(enable);
}

void CloudViewerScene::ShowAxes(bool enable) {
    auto scene = renderer_.GetScene(scene_);
    if (enable && axis_dirty_) {
        RecreateAxis(scene, bounds_, false);
        axis_dirty_ = false;
    }
    scene->ShowGeometry(kAxisObjectName, enable);
}

void CloudViewerScene::SetBackground(
        const Eigen::Vector4f& color,
        std::shared_ptr<geometry::Image> image /*=0*/) {
    auto scene = renderer_.GetScene(scene_);
    scene->SetBackground(color, image);
    background_color = color;
}

const Eigen::Vector4f CloudViewerScene::GetBackgroundColor() const {
    return background_color;
}

void CloudViewerScene::ShowGroundPlane(bool enable, Scene::GroundPlane plane) {
    auto scene = renderer_.GetScene(scene_);
    scene->EnableGroundPlane(enable, plane);
}

void CloudViewerScene::SetLighting(LightingProfile profile,
                                   const Eigen::Vector3f& sun_dir) {
    auto scene = renderer_.GetScene(scene_);

    if (profile != LightingProfile::HARD_SHADOWS) {
        if (scene->GetIndirectLight().empty()) {
            auto path = gui::Application::GetInstance().GetResourcePath();
            scene->SetIndirectLight(std::string(path) + "/default");
        }
    }

    Eigen::Vector3f sun_color(1.0f, 1.0f, 1.0f);

    // These intensities have been chosen so that a white object on a white
    // background is clearly visible even when the highlight is next to the
    // background. Increasing the intensities much more make the highlight's
    // white too similar to the background's white.
    switch (profile) {
        case LightingProfile::HARD_SHADOWS:
            scene->EnableIndirectLight(false);
            scene->EnableSunLight(true);
            scene->SetSunLight(sun_dir, sun_color, 100000);
            break;
        case LightingProfile::DARK_SHADOWS:
            scene->EnableIndirectLight(true);
            scene->EnableSunLight(true);
            scene->SetIndirectLightIntensity(5000);
            scene->SetSunLight(sun_dir, sun_color, 85000);
            break;
        default:
        case LightingProfile::MED_SHADOWS:
            scene->EnableIndirectLight(true);
            scene->EnableSunLight(true);
            scene->SetIndirectLightIntensity(7500);
            scene->SetSunLight(sun_dir, sun_color, 70000);
            break;
        case LightingProfile::SOFT_SHADOWS:
            scene->EnableIndirectLight(true);
            scene->EnableSunLight(true);
            scene->SetIndirectLightIntensity(15000);
            scene->SetSunLight(sun_dir, sun_color, 35000);
            break;
        case LightingProfile::NO_SHADOWS:
            scene->EnableIndirectLight(true);
            scene->SetIndirectLightIntensity(20000);
            scene->EnableSunLight(false);
            break;
    }
}

void CloudViewerScene::ClearGeometry() {
    auto scene = renderer_.GetScene(scene_);
    for (auto& g : geometries_) {
        scene->RemoveGeometry(g.second.name);
        if (!g.second.fast_name.empty()) {
            scene->RemoveGeometry(g.second.fast_name);
        }
        if (!g.second.low_name.empty()) {
            scene->RemoveGeometry(g.second.low_name);
        }
    }
    geometries_.clear();
    bounds_ = ccBBox();
    axis_dirty_ = true;
}

void CloudViewerScene::AddGeometry(
        const std::string& name,
        const ccHObject* geom,
        const MaterialRecord& mat,
        bool add_downsampled_copy_for_fast_rendering /*= true*/) {
    size_t downsample_threshold = SIZE_MAX;
    std::string fast_name;
    if (add_downsampled_copy_for_fast_rendering) {
        fast_name = name + "." + kFastModelObjectSuffix;
        downsample_threshold = downsample_threshold_;
    }

    auto scene = renderer_.GetScene(scene_);
    if (scene->AddGeometry(name, *geom, mat, fast_name, downsample_threshold)) {
        bounds_ += scene->GetGeometryBoundingBox(name);
        GeometryData info(name, "");
        // If the downsampled object got created, add it. It may not have been
        // created if downsampling wasn't enabled or if the object does not meet
        // the threshold.
        if (add_downsampled_copy_for_fast_rendering &&
            scene->HasGeometry(fast_name)) {
            info.fast_name = fast_name;
            geometries_[fast_name] = info;
        }
        geometries_[name] = info;
        SetGeometryToLOD(info, lod_);
    }

    axis_dirty_ = true;
}

void CloudViewerScene::AddGeometry(
        const std::string& name,
        const t::geometry::PointCloud* geom,
        const MaterialRecord& mat,
        bool add_downsampled_copy_for_fast_rendering /*= true*/) {
    size_t downsample_threshold = SIZE_MAX;
    std::string fast_name;
    if (add_downsampled_copy_for_fast_rendering) {
        fast_name = name + "." + kFastModelObjectSuffix;
        downsample_threshold = downsample_threshold_;
    }

    auto scene = renderer_.GetScene(scene_);
    if (scene->AddGeometry(name, *geom, mat, fast_name, downsample_threshold)) {
        auto bbox = scene->GetGeometryBoundingBox(name);
        bounds_ += bbox;
        GeometryData info(name, "");
        // If the downsampled object got created, add it. It may not have been
        // created if downsampling wasn't enabled or if the object does not meet
        // the threshold.
        if (add_downsampled_copy_for_fast_rendering &&
            scene->HasGeometry(fast_name)) {
            info.fast_name = fast_name;

            auto lowq_name = name + kLowQualityModelObjectSuffix;
            auto bbox_geom =
                    geometry::LineSet::CreateFromAxisAlignedBoundingBox(bbox);
            MaterialRecord bbox_mat;
            bbox_mat.base_color = {1.0f, 0.5f, 0.0f, 1.0f};  // orange
            bbox_mat.shader = "unlitSolidColor";
            scene->AddGeometry(lowq_name, *bbox_geom, bbox_mat);
            info.low_name = lowq_name;
        }
        geometries_[name] = info;
        SetGeometryToLOD(info, lod_);
    }

    // Axes may need to be recreated
    axis_dirty_ = true;
}

bool CloudViewerScene::HasGeometry(const std::string& name) const {
    auto scene = renderer_.GetScene(scene_);
    return scene->HasGeometry(name);
}

void CloudViewerScene::RemoveGeometry(const std::string& name) {
    auto scene = renderer_.GetScene(scene_);
    auto g = geometries_.find(name);
    if (g != geometries_.end()) {
        scene->RemoveGeometry(name);
        if (!g->second.fast_name.empty()) {
            scene->RemoveGeometry(g->second.fast_name);
        }
        if (!g->second.low_name.empty()) {
            scene->RemoveGeometry(g->second.low_name);
        }
        geometries_.erase(name);
    }
}

void CloudViewerScene::ModifyGeometryMaterial(const std::string& name,
                                              const MaterialRecord& mat) {
    auto scene = renderer_.GetScene(scene_);
    scene->OverrideMaterial(name, mat);
    auto it = geometries_.find(name);
    if (it != geometries_.end()) {
        if (!it->second.fast_name.empty()) {
            scene->OverrideMaterial(it->second.fast_name, mat);
        }
        // Don't want to override low_name, as that is a bounding box.
    }
}

void CloudViewerScene::ShowGeometry(const std::string& name, bool show) {
    auto it = geometries_.find(name);
    if (it != geometries_.end()) {
        it->second.visible = show;

        int n_lowq_visible = 0;
        for (auto& g : geometries_) {
            if (g.second.visible && !g.second.low_name.empty()) {
                n_lowq_visible += 1;
            }
        }
        use_low_quality_if_available_ = (n_lowq_visible > 1);

        SetGeometryToLOD(it->second, lod_);
    }
}

void CloudViewerScene::AddModel(const std::string& name,
                                const TriangleMeshModel& model) {
    auto scene = renderer_.GetScene(scene_);
    if (scene->AddGeometry(name, model)) {
        GeometryData info(name, "");
        bounds_ += scene->GetGeometryBoundingBox(name);
        geometries_[name] = info;
        scene->ShowGeometry(name, true);
    }

    axis_dirty_ = true;
}

void CloudViewerScene::UpdateMaterial(const MaterialRecord& mat) {
    auto scene = renderer_.GetScene(scene_);
    for (auto& g : geometries_) {
        scene->OverrideMaterial(g.second.name, mat);
        if (!g.second.fast_name.empty()) {
            scene->OverrideMaterial(g.second.fast_name, mat);
        }
        // Low-quality model is a bounding box right now, and we want it to
        // be a solid color, so we do not want to override.
        // if (!g.second.low_name.empty()) {
        //     scene->OverrideMaterial(g.second.low_name, mat);
        // }
    }
}

void CloudViewerScene::UpdateModelMaterial(const std::string& name,
                                           const TriangleMeshModel& model) {
    auto scene = renderer_.GetScene(scene_);
    scene->RemoveGeometry(name);
    scene->AddGeometry(name, model);
}

std::vector<std::string> CloudViewerScene::GetGeometries() {
    std::vector<std::string> names;
    names.reserve(geometries_.size());
    for (auto& it : geometries_) {
        names.push_back(it.first);
    }
    return names;
}

void CloudViewerScene::SetLOD(LOD lod) {
    if (lod != lod_) {
        lod_ = lod;

        for (auto& g : geometries_) {
            SetGeometryToLOD(g.second, lod);
        }
    }
}

void CloudViewerScene::SetGeometryToLOD(const GeometryData& data, LOD lod) {
    auto scene = renderer_.GetScene(scene_);
    scene->ShowGeometry(data.name, false);
    if (!data.fast_name.empty()) {
        scene->ShowGeometry(data.fast_name, false);
    }
    if (!data.low_name.empty()) {
        scene->ShowGeometry(data.low_name, false);
    }

    if (data.visible) {
        if (lod == LOD::HIGH_DETAIL) {
            scene->ShowGeometry(data.name, true);
        } else {
            std::string id;
            if (use_low_quality_if_available_) {
                id = data.low_name;
            }
            if (id.empty()) {
                id = data.fast_name;
            }
            if (id.empty()) {
                id = data.name;
            }
            scene->ShowGeometry(id, true);
        }
    }
}

CloudViewerScene::LOD CloudViewerScene::GetLOD() const { return lod_; }

Scene* CloudViewerScene::GetScene() const { return renderer_.GetScene(scene_); }

Camera* CloudViewerScene::GetCamera() const {
    auto scene = renderer_.GetScene(scene_);
    auto view = scene->GetView(view_);
    return view->GetCamera();
}

Renderer& CloudViewerScene::GetRenderer() const { return renderer_; }

}  // namespace rendering
}  // namespace visualization
}  // namespace cloudViewer
