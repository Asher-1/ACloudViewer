// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file Cc2Vtk.cpp
 * @brief Implementation of CloudViewer to VTK data conversion.
 */

#include "Cc2Vtk.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvMaterialSet.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

// CV_CORE_LIB
#include <CVLog.h>

// VTK
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkMatrix4x4.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkStringArray.h>
#include <vtkUnsignedCharArray.h>

// LineSet
#include <LineSet.h>

// System
#include <cassert>

namespace Converters {

vtkSmartPointer<vtkPolyData> Cc2Vtk::PointCloudToPolyData(
        const ccPointCloud* cloud,
        bool include_colors,
        bool include_normals,
        bool include_sf,
        bool show_mode) {
    if (!cloud || cloud->size() == 0) {
        return nullptr;
    }

    const unsigned point_count = cloud->size();
    const auto& visibility = cloud->getTheVisibilityArray();
    const bool partial_visibility =
            show_mode && (visibility.size() >= cloud->size());

    unsigned visible_count = point_count;
    if (partial_visibility) {
        visible_count = 0;
        for (unsigned i = 0; i < point_count; ++i) {
            if (visibility.at(i) == POINT_VISIBLE) ++visible_count;
        }
    }

    if (visible_count == 0) return nullptr;

    auto nr = static_cast<vtkIdType>(visible_count);

    // Points
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetDataTypeToFloat();
    points->SetNumberOfPoints(nr);
    float* point_data =
            static_cast<vtkFloatArray*>(points->GetData())->GetPointer(0);

    // Colors (RGB scalars)
    vtkSmartPointer<vtkUnsignedCharArray> colors;
    bool has_valid_sf = include_sf && cloud->hasScalarFields() &&
                        cloud->getCurrentDisplayedScalarFieldIndex() >= 0;
    bool want_colors = include_colors && (cloud->hasColors() || has_valid_sf);

    CVLog::PrintDebug(
            "[Cc2Vtk::PointCloudToPolyData] include_sf=%d "
            "hasDisplayedSF=%d hasScalarFields=%d sfIdx=%d "
            "hasColors=%d want_colors=%d has_valid_sf=%d",
            include_sf ? 1 : 0, cloud->hasDisplayedScalarField() ? 1 : 0,
            cloud->hasScalarFields() ? 1 : 0,
            cloud->getCurrentDisplayedScalarFieldIndex(),
            cloud->hasColors() ? 1 : 0, want_colors ? 1 : 0,
            has_valid_sf ? 1 : 0);

    if (want_colors) {
        colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetNumberOfComponents(3);
        colors->SetNumberOfTuples(nr);
        colors->SetName("RGB");
    }

    // Normals
    vtkSmartPointer<vtkFloatArray> normals;
    if (include_normals && cloud->hasNormals()) {
        normals = vtkSmartPointer<vtkFloatArray>::New();
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(nr);
        normals->SetName("Normals");
    }

    cloudViewer::ScalarField* active_sf = nullptr;
    if (has_valid_sf) {
        int sf_idx = cloud->getCurrentDisplayedScalarFieldIndex();
        active_sf = cloud->getScalarField(sf_idx);
    }
    const bool use_sf_colors = has_valid_sf && active_sf != nullptr;
    const bool use_rgb_colors = !use_sf_colors && cloud->hasColors();

    // Fill data
    vtkIdType idx = 0;
    for (unsigned i = 0; i < point_count; ++i) {
        if (partial_visibility && visibility.at(i) != POINT_VISIBLE) continue;

        const CCVector3* p = cloud->getPoint(i);
        vtkIdType base = idx * 3;
        point_data[base] = static_cast<float>(p->x);
        point_data[base + 1] = static_cast<float>(p->y);
        point_data[base + 2] = static_cast<float>(p->z);

        if (colors) {
            unsigned char* c = colors->GetPointer(base);
            if (use_sf_colors) {
                const ecvColor::Rgb* rgb =
                        cloud->getScalarValueColor(active_sf->getValue(i));
                if (rgb) {
                    c[0] = rgb->r;
                    c[1] = rgb->g;
                    c[2] = rgb->b;
                }
            } else if (use_rgb_colors) {
                const ecvColor::Rgb& rgb = cloud->getPointColor(i);
                c[0] = rgb.r;
                c[1] = rgb.g;
                c[2] = rgb.b;
            }
        }

        if (normals) {
            const CCVector3& n = cloud->getPointNormal(i);
            float* nptr = normals->GetPointer(base);
            nptr[0] = static_cast<float>(n.x);
            nptr[1] = static_cast<float>(n.y);
            nptr[2] = static_cast<float>(n.z);
        }

        ++idx;
    }

    // Vertex cells (one cell per point)
    auto vertices = vtkSmartPointer<vtkCellArray>::New();
    for (vtkIdType i = 0; i < nr; ++i) {
        vertices->InsertNextCell(1);
        vertices->InsertCellPoint(i);
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetVerts(vertices);

    if (colors) {
        polydata->GetPointData()->SetScalars(colors);
    }
    if (normals) {
        polydata->GetPointData()->SetNormals(normals);
    }

    // Add HasSourceRGB flag for Vtk2Cc reverse conversion
    if (want_colors && cloud->hasColors() && !include_sf) {
        auto flag = vtkSmartPointer<vtkStringArray>::New();
        flag->SetName("HasSourceRGB");
        flag->SetNumberOfTuples(1);
        flag->SetValue(0, "1");
        polydata->GetFieldData()->AddArray(flag);
    }

    // DatasetName for ParaView-style tooltip
    QString name = cloud->getName();
    if (!name.isEmpty()) {
        auto ds = vtkSmartPointer<vtkStringArray>::New();
        ds->SetName("DatasetName");
        ds->SetNumberOfTuples(1);
        ds->SetValue(0, name.toStdString());
        polydata->GetFieldData()->AddArray(ds);
    }

    return polydata;
}

bool Cc2Vtk::GetVtkScalars(const ccPointCloud* cloud,
                           vtkSmartPointer<vtkDataArray>& scalars,
                           bool sf_colors,
                           bool show_mode) {
    if (!cloud) return false;

    const auto& visibility = cloud->getTheVisibilityArray();
    const bool partial = show_mode && (visibility.size() >= cloud->size());
    const unsigned point_count = cloud->size();

    unsigned nr_points = point_count;
    if (partial) {
        nr_points = 0;
        for (unsigned i = 0; i < point_count; ++i) {
            if (visibility.at(i) == POINT_VISIBLE) ++nr_points;
        }
    }

    if (!scalars) scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    auto* char_array = static_cast<vtkUnsignedCharArray*>(scalars.Get());
    char_array->SetNumberOfTuples(static_cast<vtkIdType>(nr_points));
    unsigned char* colors = char_array->GetPointer(0);

    int j = 0;
    if (cloud->hasScalarFields() && sf_colors) {
        int sf_idx = cloud->getCurrentDisplayedScalarFieldIndex();
        if (sf_idx < 0) return false;
        auto* sf = cloud->getScalarField(sf_idx);
        if (!sf) return false;
        for (unsigned cp = 0; cp < point_count; ++cp) {
            if (partial && visibility.at(cp) != POINT_VISIBLE) continue;
            const ecvColor::Rgb* rgb =
                    cloud->getScalarValueColor(sf->getValue(cp));
            colors[j] = rgb->r;
            colors[j + 1] = rgb->g;
            colors[j + 2] = rgb->b;
            j += 3;
        }
    } else if (cloud->hasColors()) {
        for (unsigned cp = 0; cp < point_count; ++cp) {
            if (partial && visibility.at(cp) != POINT_VISIBLE) continue;
            const ecvColor::Rgb& rgb = cloud->getPointColor(cp);
            colors[j] = rgb.r;
            colors[j + 1] = rgb.g;
            colors[j + 2] = rgb.b;
            j += 3;
        }
    } else {
        return false;
    }

    return true;
}

vtkSmartPointer<vtkPolyData> Cc2Vtk::MeshToPolyData(
        const ccPointCloud* vertex_cloud, ccGenericMesh* mesh) {
    if (!mesh || !vertex_cloud) return nullptr;

    const unsigned tri_count = mesh->size();
    if (tri_count == 0) return nullptr;

    const std::size_t dim = static_cast<std::size_t>(
            mesh->getTriangleVertIndexes(0)->getDimension());

    const auto& vis = vertex_cloud->getTheVisibilityArray();
    const bool vis_filter = (vis.size() >= vertex_cloud->size());

    const bool show_sf = mesh->hasDisplayedScalarField() && mesh->sfShown();
    const bool show_colors =
            show_sf || (mesh->hasColors() && mesh->colorsShown());
    const bool has_normals_export = mesh->hasTriNormals() || mesh->hasNormals();

    const std::size_t total_pts = static_cast<std::size_t>(tri_count) * dim;
    const auto total_vtk = static_cast<vtkIdType>(total_pts);

    auto poly_points = vtkSmartPointer<vtkPoints>::New();
    poly_points->SetNumberOfPoints(total_vtk);

    vtkSmartPointer<vtkUnsignedCharArray> colors;
    if (show_colors) {
        colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetNumberOfComponents(3);
        colors->SetNumberOfTuples(total_vtk);
        colors->SetName("RGB");
    }

    vtkSmartPointer<vtkFloatArray> normals;
    if (has_normals_export) {
        normals = vtkSmartPointer<vtkFloatArray>::New();
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(total_vtk);
        normals->SetName("Normals");
    }

    auto polys = vtkSmartPointer<vtkCellArray>::New();
    polys->AllocateEstimate(static_cast<vtkIdType>(tri_count), 3);

    const NormsIndexesTableType* tri_norms = mesh->getTriNormsTable();
    NormsIndexesTableType* norms_table = nullptr;
    ccNormalVectors* compressed_norms = nullptr;
    if (has_normals_export) {
        norms_table = vertex_cloud->normals();
        compressed_norms = ccNormalVectors::GetUniqueInstance();
    }

    for (unsigned n = 0; n < tri_count; ++n) {
        const cloudViewer::VerticesIndexes* tsi =
                mesh->getTriangleVertIndexes(n);
        if (vis_filter) {
            if (vis[tsi->i1] != POINT_VISIBLE ||
                vis[tsi->i2] != POINT_VISIBLE || vis[tsi->i3] != POINT_VISIBLE)
                continue;
        }

        auto base = static_cast<vtkIdType>(n) * static_cast<vtkIdType>(dim);

        const PointCoordinateType* N1 = nullptr;
        const PointCoordinateType* N2 = nullptr;
        const PointCoordinateType* N3 = nullptr;

        if (has_normals_export) {
            bool use_tri = mesh->hasTriNormals() && tri_norms;
            if (use_tri) {
                int n1 = 0, n2 = 0, n3 = 0;
                mesh->getTriangleNormalIndexes(n, n1, n2, n3);
                N1 = (n1 >= 0 ? ccNormalVectors::GetNormal(tri_norms->at(n1)).u
                              : nullptr);
                N2 = (n1 == n2 ? N1
                      : n2 >= 0
                              ? ccNormalVectors::GetNormal(tri_norms->at(n2)).u
                              : nullptr);
                N3 = (n1 == n3 ? N1
                      : n3 >= 0
                              ? ccNormalVectors::GetNormal(tri_norms->at(n3)).u
                              : nullptr);
            } else if (norms_table && compressed_norms) {
                N1 = compressed_norms->getNormal(norms_table->at(tsi->i1)).u;
                N2 = compressed_norms->getNormal(norms_table->at(tsi->i2)).u;
                N3 = compressed_norms->getNormal(norms_table->at(tsi->i3)).u;
            }
        }

        for (std::size_t vi = 0; vi < dim; ++vi) {
            auto pt_idx = base + static_cast<vtkIdType>(vi);
            unsigned vert_idx = tsi->i[vi];

            const CCVector3* p = vertex_cloud->getPoint(vert_idx);
            poly_points->SetPoint(pt_idx, p->x, p->y, p->z);

            if (colors) {
                const ecvColor::Rgb* rgb = nullptr;
                if (show_sf) {
                    rgb = vertex_cloud->getCurrentDisplayedScalarField()
                                  ->getValueColor(vert_idx);
                } else {
                    rgb = &vertex_cloud->rgbColors()->at(vert_idx);
                }
                unsigned char* c = colors->GetPointer(pt_idx * 3);
                c[0] = rgb->r;
                c[1] = rgb->g;
                c[2] = rgb->b;
            }

            if (normals) {
                const PointCoordinateType* N =
                        (vi == 0 ? N1 : (vi == 1 ? N2 : N3));
                float* nptr = normals->GetPointer(pt_idx * 3);
                if (N) {
                    nptr[0] = static_cast<float>(N[0]);
                    nptr[1] = static_cast<float>(N[1]);
                    nptr[2] = static_cast<float>(N[2]);
                } else {
                    nptr[0] = nptr[1] = nptr[2] = 0.0f;
                }
            }
        }

        polys->InsertNextCell(3);
        polys->InsertCellPoint(base + 0);
        polys->InsertCellPoint(base + 1);
        polys->InsertCellPoint(base + 2);
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(poly_points);
    polydata->SetPolys(polys);

    if (colors) {
        if (show_sf && mesh->hasDisplayedScalarField()) {
            auto* pc = ccHObjectCaster::ToPointCloud(
                    const_cast<ccGenericPointCloud*>(
                            mesh->getAssociatedCloud()));
            if (pc) {
                int sf_idx = pc->getCurrentDisplayedScalarFieldIndex();
                if (sf_idx >= 0)
                    colors->SetName(pc->getScalarFieldName(sf_idx));
            }
        }
        polydata->GetPointData()->SetScalars(colors);
    }
    if (normals) {
        polydata->GetPointData()->SetNormals(normals);
    }

    QString mesh_name = mesh->getName();
    if (!mesh_name.isEmpty()) {
        auto ds = vtkSmartPointer<vtkStringArray>::New();
        ds->SetName("DatasetName");
        ds->SetNumberOfTuples(1);
        ds->SetValue(0, mesh_name.toStdString());
        polydata->GetFieldData()->AddArray(ds);
    }

    return polydata;
}

bool Cc2Vtk::TextureMeshToPolyData(
        const ccPointCloud* vertex_cloud,
        ccGenericMesh* mesh,
        vtkSmartPointer<vtkPolyData>& polydata,
        vtkSmartPointer<vtkMatrix4x4>& transformation,
        std::vector<std::vector<Eigen::Vector2f>>& tex_coordinates) {
    if (!mesh || !vertex_cloud) return false;

    polydata = MeshToPolyData(vertex_cloud, mesh);
    if (!polydata) return false;

    bool apply_materials = mesh->hasMaterials() && mesh->materialsShown();
    bool show_textures = mesh->hasTextures() && mesh->materialsShown();
    if (!apply_materials && !show_textures) return false;

    const unsigned tri_count = mesh->size();
    if (tri_count == 0) return false;

    const std::size_t dim = static_cast<std::size_t>(
            mesh->getTriangleVertIndexes(0)->getDimension());

    const auto& vis = mesh->getAssociatedCloud()->getTheVisibilityArray();
    const bool vis_filter = (vis.size() >= mesh->getAssociatedCloud()->size());

    const ccMaterialSet* materials = mesh->getMaterialSet();
    assert(materials);

    std::vector<std::vector<Eigen::Vector2f>> mat_tex_coords;
    std::vector<int> tri_to_mat;
    tri_to_mat.reserve(tri_count);

    int last_mat = -1;
    int current_mat = -1;

    for (unsigned n = 0; n < tri_count; ++n) {
        const cloudViewer::VerticesIndexes* tsi =
                mesh->getTriangleVertIndexes(n);
        if (vis_filter) {
            if (vis[tsi->i1] != POINT_VISIBLE ||
                vis[tsi->i2] != POINT_VISIBLE || vis[tsi->i3] != POINT_VISIBLE)
                continue;
        }

        int new_mat = mesh->getTriangleMtlIndex(n);
        if (last_mat != new_mat) {
            current_mat = static_cast<int>(mat_tex_coords.size());
            mat_tex_coords.emplace_back();
            last_mat = new_mat;
        }

        if (n >= tri_to_mat.size()) tri_to_mat.resize(n + 1, -1);
        tri_to_mat[n] = current_mat;

        if (show_textures && current_mat >= 0) {
            TexCoords2D* tx1 = nullptr;
            TexCoords2D* tx2 = nullptr;
            TexCoords2D* tx3 = nullptr;
            mesh->getTriangleTexCoordinates(n, tx1, tx2, tx3);
            if (tx1 && tx2 && tx3) {
                mat_tex_coords[current_mat].emplace_back(tx1->tx, tx1->ty);
                mat_tex_coords[current_mat].emplace_back(tx2->tx, tx2->ty);
                mat_tex_coords[current_mat].emplace_back(tx3->tx, tx3->ty);
            } else {
                for (int k = 0; k < 3; ++k)
                    mat_tex_coords[current_mat].emplace_back(0.0f, 0.0f);
            }
        } else if (current_mat >= 0) {
            for (int k = 0; k < 3; ++k)
                mat_tex_coords[current_mat].emplace_back(0.0f, 0.0f);
        }
    }

    vtkIdType num_pts = polydata->GetNumberOfPoints();
    tex_coordinates.clear();
    tex_coordinates.resize(mat_tex_coords.size());

    std::vector<size_t> mat_coord_idx(mat_tex_coords.size(), 0);

    for (vtkIdType pt = 0; pt < num_pts; ++pt) {
        int tri_idx = static_cast<int>(pt) / static_cast<int>(dim);
        int mat_idx = (tri_idx < static_cast<int>(tri_to_mat.size()))
                              ? tri_to_mat[tri_idx]
                              : -1;

        for (size_t m = 0; m < mat_tex_coords.size(); ++m) {
            if (static_cast<int>(m) == mat_idx &&
                mat_coord_idx[m] < mat_tex_coords[m].size()) {
                tex_coordinates[m].push_back(
                        mat_tex_coords[m][mat_coord_idx[m]]);
                mat_coord_idx[m]++;
            } else {
                tex_coordinates[m].emplace_back(-1.0f, -1.0f);
            }
        }
    }

    if (materials && materials->size() > 0) {
        QString mtl_file;
        QVariant mtl_data = mesh->getMetaData("MTL_FILENAME");
        if (mtl_data.isValid() && !mtl_data.toString().isEmpty())
            mtl_file = mtl_data.toString();
        else
            mtl_file = materials->at(0)->getName();

        auto lib_arr = vtkSmartPointer<vtkStringArray>::New();
        lib_arr->SetName("MaterialLibraries");
        lib_arr->SetNumberOfTuples(1);
        lib_arr->SetValue(0, mtl_file.toStdString());
        polydata->GetFieldData()->AddArray(lib_arr);

        auto names_arr = vtkSmartPointer<vtkStringArray>::New();
        names_arr->SetName("MaterialNames");
        names_arr->SetNumberOfComponents(1);
        names_arr->SetNumberOfTuples(materials->size());
        for (size_t i = 0; i < materials->size(); ++i)
            names_arr->SetValue(i, materials->at(i)->getName().toStdString());
        polydata->GetFieldData()->AddArray(names_arr);
    }

    transformation = vtkSmartPointer<vtkMatrix4x4>::New();
    transformation->Identity();
    return true;
}

vtkSmartPointer<vtkPolyData> Cc2Vtk::PolylineToPolyData(
        const ccPolyline* polyline) {
    if (!polyline || polyline->size() < 2) return nullptr;

    const unsigned count = polyline->size();
    auto points = vtkSmartPointer<vtkPoints>::New();
    points->SetNumberOfPoints(static_cast<vtkIdType>(count));

    for (unsigned i = 0; i < count; ++i) {
        const CCVector3* pp = polyline->getPoint(i);
        CCVector3d out;
        if (polyline->is2DMode()) {
            ecvDisplayTools::TheInstance()->toWorldPoint(*pp, out);
        } else {
            out = CCVector3d::fromArray(pp->u);
        }
        points->SetPoint(static_cast<vtkIdType>(i), out.x, out.y, out.z);
    }

    auto lines = vtkSmartPointer<vtkCellArray>::New();
    bool closed = polyline->isClosed();
    vtkIdType seg_count = closed ? static_cast<vtkIdType>(count)
                                 : static_cast<vtkIdType>(count) - 1;
    for (vtkIdType i = 0; i < seg_count; ++i) {
        lines->InsertNextCell(2);
        lines->InsertCellPoint(i);
        lines->InsertCellPoint((i + 1) % static_cast<vtkIdType>(count));
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetLines(lines);
    return polydata;
}

vtkSmartPointer<vtkPolyData> Cc2Vtk::LineSetToPolyData(
        const cloudViewer::geometry::LineSet* lineset) {
    if (!lineset) return nullptr;

    const auto& pts = lineset->points_;
    const auto& lns = lineset->lines_;
    if (pts.empty() || lns.empty()) return nullptr;

    auto vtk_points = vtkSmartPointer<vtkPoints>::New();
    vtk_points->SetNumberOfPoints(static_cast<vtkIdType>(pts.size()));
    for (size_t i = 0; i < pts.size(); ++i) {
        vtk_points->SetPoint(static_cast<vtkIdType>(i), pts[i].x(), pts[i].y(),
                             pts[i].z());
    }

    auto vtk_lines = vtkSmartPointer<vtkCellArray>::New();
    for (size_t i = 0; i < lns.size(); ++i) {
        vtk_lines->InsertNextCell(2);
        vtk_lines->InsertCellPoint(static_cast<vtkIdType>(lns[i].x()));
        vtk_lines->InsertCellPoint(static_cast<vtkIdType>(lns[i].y()));
    }

    vtkSmartPointer<vtkUnsignedCharArray> colors;
    if (!lineset->colors_.empty() && lineset->colors_.size() == lns.size()) {
        colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
        colors->SetNumberOfComponents(3);
        colors->SetNumberOfTuples(static_cast<vtkIdType>(lns.size()));
        for (size_t i = 0; i < lns.size(); ++i) {
            const auto& c = lineset->colors_[i];
            unsigned char rgb[3] = {static_cast<unsigned char>(c.x() * 255),
                                    static_cast<unsigned char>(c.y() * 255),
                                    static_cast<unsigned char>(c.z() * 255)};
            colors->SetTypedTuple(static_cast<vtkIdType>(i), rgb);
        }
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(vtk_points);
    polydata->SetLines(vtk_lines);
    if (colors) {
        polydata->GetCellData()->SetScalars(colors);
    }
    return polydata;
}

void Cc2Vtk::AddScalarFieldToPolyData(vtkPolyData* polydata,
                                      const ccPointCloud* cloud,
                                      int scalar_field_index) {
    if (!polydata || !cloud) return;
    if (scalar_field_index < 0 ||
        scalar_field_index >=
                static_cast<int>(cloud->getNumberOfScalarFields()))
        return;

    auto* sf = cloud->getScalarField(scalar_field_index);
    if (!sf) return;

    std::string name = GetSimplifiedSFName(sf->getName());
    vtkIdType n_pts = polydata->GetNumberOfPoints();
    vtkIdType sf_count = static_cast<vtkIdType>(sf->size());

    if (n_pts != sf_count) {
        CVLog::Warning(
                QString("[Cc2Vtk::AddScalarFieldToPolyData] Size mismatch: "
                        "polydata=%1, SF=%2")
                        .arg(n_pts)
                        .arg(sf_count));
        return;
    }

    auto existing = polydata->GetPointData()->GetArray(name.c_str());
    if (existing) return;

    auto arr = vtkSmartPointer<vtkFloatArray>::New();
    arr->SetName(name.c_str());
    arr->SetNumberOfComponents(1);
    arr->SetNumberOfTuples(n_pts);
    for (vtkIdType i = 0; i < n_pts; ++i) {
        arr->SetValue(i, sf->getValue(static_cast<unsigned>(i)));
    }
    polydata->GetPointData()->AddArray(arr);

    auto ds_name_arr = vtkStringArray::SafeDownCast(
            polydata->GetFieldData()->GetAbstractArray("DatasetName"));
    if (!ds_name_arr) {
        QString cloud_name = cloud->getName();
        if (!cloud_name.isEmpty()) {
            auto ds = vtkSmartPointer<vtkStringArray>::New();
            ds->SetName("DatasetName");
            ds->SetNumberOfTuples(1);
            ds->SetValue(0, cloud_name.toStdString());
            polydata->GetFieldData()->AddArray(ds);
        }
    }
}

std::string Cc2Vtk::GetSimplifiedSFName(const std::string& cc_sf_name) {
    QString simplified = QString::fromStdString(cc_sf_name).simplified();
    simplified.replace(' ', '_');
    return simplified.toStdString();
}

}  // namespace Converters
