// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "sm2cc.h"

// Local
#include "PCLConv.h"
#include "my_point_types.h"

// PCL
#include <pcl/common/io.h>

// CV_CORE_LIB
#include <CVTools.h>
#include <FileSystem.h>

// ECV_DB_LIB
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// system
#include <assert.h>

#include <list>

// Qt
#include <pcl/PCLPointField.h>

#include <QImageReader>
typedef pcl::PCLPointField PCLScalarField;

// Custom PCL point types
template <typename T>
struct PointXYZTpl {
    union EIGEN_ALIGN16 {
        T data[3];
        struct {
            T x;
            T y;
            T z;
        };
    };
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTpl<std::int32_t>,
                                  (std::int32_t, x, x)(std::int32_t,
                                                       y,
                                                       y)(std::int32_t, z, z))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTpl<std::int16_t>,
                                  (std::int16_t, x, x)(std::int16_t,
                                                       y,
                                                       y)(std::int16_t, z, z))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZTpl<double>,
                                  (double, x, x)(double, y, y)(double, z, z))

size_t GetNumberOfPoints(const PCLCloud& pclCloud) {
    return static_cast<size_t>(pclCloud.width) * pclCloud.height;
}

bool ExistField(const PCLCloud& pclCloud, std::string name) {
    for (const auto& field : pclCloud.fields)
        if (field.name == name) return true;

    return false;
}

template <class T>
void PCLCloudToCCCloud(const PCLCloud& pclCloud, ccPointCloud& ccCloud) {
    size_t pointCount = GetNumberOfPoints(pclCloud);

    pcl::PointCloud<T> pcl_cloud;
    FROM_PCL_CLOUD(pclCloud, pcl_cloud);
    for (size_t i = 0; i < pointCount; ++i) {
        CCVector3 P(pcl_cloud.at(i).x, pcl_cloud.at(i).y, pcl_cloud.at(i).z);

        ccCloud.addPoint(P);
    }
}

bool pcl2cc::CopyXYZ(const PCLCloud& pclCloud,
                     ccPointCloud& ccCloud,
                     uint8_t coordinateType) {
    size_t pointCount = GetNumberOfPoints(pclCloud);
    if (pointCount == 0) {
        assert(false);
        return false;
    }

    if (!ccCloud.reserve(static_cast<unsigned>(pointCount))) {
        return false;
    }

    // add xyz to the input cloud taking xyz infos from the sm cloud
    switch (coordinateType) {
        case pcl::PCLPointField::INT16:
            PCLCloudToCCCloud<PointXYZTpl<std::int16_t>>(pclCloud, ccCloud);
            break;
        case pcl::PCLPointField::INT32:
            PCLCloudToCCCloud<PointXYZTpl<std::int32_t>>(pclCloud, ccCloud);
            break;
        case pcl::PCLPointField::FLOAT32:
            PCLCloudToCCCloud<pcl::PointXYZ>(pclCloud, ccCloud);
            break;
        case pcl::PCLPointField::FLOAT64:
            PCLCloudToCCCloud<PointXYZTpl<double>>(pclCloud, ccCloud);
            break;
        default:
            CVLog::Warning("[PCL] Unsupported coordinate type " +
                           QString::number(coordinateType));
            return false;
    };

    return true;
}

bool pcl2cc::CopyNormals(const PCLCloud& pclCloud, ccPointCloud& ccCloud) {
    pcl::PointCloud<OnlyNormals>::Ptr pcl_cloud_normals(
            new pcl::PointCloud<OnlyNormals>);
    FROM_PCL_CLOUD(pclCloud, *pcl_cloud_normals);

    if (!ccCloud.reserveTheNormsTable()) return false;

    size_t pointCount = GetNumberOfPoints(pclCloud);

    // loop
    for (size_t i = 0; i < pointCount; ++i) {
        CCVector3 N(static_cast<PointCoordinateType>(
                            pcl_cloud_normals->at(i).normal_x),
                    static_cast<PointCoordinateType>(
                            pcl_cloud_normals->at(i).normal_y),
                    static_cast<PointCoordinateType>(
                            pcl_cloud_normals->at(i).normal_z));

        ccCloud.addNorm(N);
    }

    ccCloud.showNormals(true);

    return true;
}

bool pcl2cc::CopyRGB(const PCLCloud& pclCloud, ccPointCloud& ccCloud) {
    pcl::PointCloud<OnlyRGB>::Ptr pcl_cloud_rgb(new pcl::PointCloud<OnlyRGB>);
    FROM_PCL_CLOUD(pclCloud, *pcl_cloud_rgb);
    size_t pointCount = GetNumberOfPoints(pclCloud);
    if (pointCount == 0) return true;
    if (!ccCloud.reserveTheRGBTable()) return false;

    // loop
    for (size_t i = 0; i < pointCount; ++i) {
        ecvColor::Rgb C(static_cast<ColorCompType>(pcl_cloud_rgb->points[i].r),
                        static_cast<ColorCompType>(pcl_cloud_rgb->points[i].g),
                        static_cast<ColorCompType>(pcl_cloud_rgb->points[i].b));
        ccCloud.addRGBColor(C);
    }

    ccCloud.showColors(true);

    return true;
}

bool pcl2cc::CopyScalarField(const PCLCloud& pclCloud,
                             const std::string& sfName,
                             ccPointCloud& ccCloud,
                             bool overwriteIfExist /*=true*/) {
    // if the input field already exists...
    int id = ccCloud.getScalarFieldIndexByName(sfName.c_str());
    if (id >= 0) {
        if (overwriteIfExist)
            // we simply delete it
            ccCloud.deleteScalarField(id);
        else
            // we keep it as is
            return false;
    }

    size_t pointCount = GetNumberOfPoints(pclCloud);

    // create new scalar field
    ccScalarField* newSF = new ccScalarField(sfName.c_str());
    if (!newSF->reserveSafe(static_cast<unsigned>(pointCount))) {
        newSF->release();
        return false;
    }

    // get PCL field
    int field_index = pcl::getFieldIndex(pclCloud, sfName);
    if (field_index < 0) {
        newSF->release();
        return false;
    }
    const PCLScalarField& pclField = pclCloud.fields[field_index];
    // temporary change the name of the given field to something else -> S5c4laR
    // should be a pretty uncommon name,
    const_cast<PCLScalarField&>(pclField).name = "S5c4laR";

    switch (pclField.datatype) {
        case PCLScalarField::FLOAT32: {
            pcl::PointCloud<FloatScalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::FLOAT64: {
            pcl::PointCloud<DoubleScalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::INT8: {
            pcl::PointCloud<Int8Scalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::UINT8: {
            pcl::PointCloud<UInt8Scalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::INT16: {
            pcl::PointCloud<ShortScalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::UINT16: {
            pcl::PointCloud<UShortScalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::UINT32: {
            pcl::PointCloud<UIntScalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        case PCLScalarField::INT32: {
            pcl::PointCloud<IntScalar> pclScalar;
            FROM_PCL_CLOUD(pclCloud, pclScalar);

            for (unsigned i = 0; i < pointCount; ++i) {
                ScalarType scalar =
                        static_cast<ScalarType>(pclScalar.points[i].S5c4laR);
                newSF->addElement(scalar);
            }
        } break;

        default:
            CVLog::Warning(QString("[PCL] Field with an unmanaged type (= %1)")
                                   .arg(pclField.datatype));
            newSF->release();
            return false;
    }
    newSF->computeMinAndMax();
    int sfIdex = ccCloud.addScalarField(newSF);
    ccCloud.setCurrentDisplayedScalarField(sfIdex);
    ccCloud.showSF(true);

    // restore old name for the scalar field
    const_cast<PCLScalarField&>(pclField).name = sfName;

    return true;
}

void pcl2cc::FromPCLMaterial(const PCLMaterial& inMaterial,
                             ccMaterial::Shared& outMaterial) {
    QString cPath = CVTools::ToQString(inMaterial.tex_file);
    QString parentPath = CVTools::ToQString(
            cloudViewer::utility::filesystem::GetFileParentDirectory(
                    inMaterial.tex_file));
    outMaterial->setName(inMaterial.tex_name.c_str());

    cPath = CVTools::ToNativeSeparators(cPath);

    if (QFile::exists(cPath)) {
        QImageReader reader(cPath);

        QImage image = reader.read();
        if (image.isNull()) {
            CVLog::Warning(
                    QString("[pcl2ccMaterial] failed to read image %1, %2")
                            .arg(cPath)
                            .arg(reader.errorString()));
        }

        if (!image.isNull()) {
            outMaterial->setTexture(image, parentPath, false);
        }
    }

    std::string texName = inMaterial.tex_name;
    // FIX special symbols bugs in vtk rendering system!
    texName = CVTools::ExtractDigitAlpha(texName);
    const ecvColor::Rgbaf ambientColor(inMaterial.tex_Ka.r, inMaterial.tex_Ka.g,
                                       inMaterial.tex_Ka.b, inMaterial.tex_d);
    const ecvColor::Rgbaf diffuseColor(inMaterial.tex_Kd.r, inMaterial.tex_Kd.g,
                                       inMaterial.tex_Kd.b, inMaterial.tex_d);
    const ecvColor::Rgbaf specularColor(inMaterial.tex_Ks.r,
                                        inMaterial.tex_Ks.g,
                                        inMaterial.tex_Ks.b, inMaterial.tex_d);
    float shininess = inMaterial.tex_Ns;

    outMaterial->setDiffuse(diffuseColor);
    outMaterial->setAmbient(ambientColor);
    outMaterial->setSpecular(specularColor);
    outMaterial->setEmission(ecvColor::night);
    outMaterial->setShininess(shininess);
    outMaterial->setTransparency(inMaterial.tex_d);
    outMaterial->setIllum(inMaterial.tex_illum);
}

ccMesh* pcl2cc::Convert(pcl::TextureMesh::ConstPtr textureMesh) {
    if (!textureMesh || textureMesh->tex_polygons.empty()) {
        return nullptr;
    }

    // mesh size
    std::size_t nr_meshes = textureMesh->tex_polygons.size();
    // number of faces for header
    std::size_t nr_faces = 0;
    for (std::size_t m = 0; m < nr_meshes; ++m) {
        nr_faces += textureMesh->tex_polygons[m].size();
    }

    // create mesh from PCLMaterial
    std::vector<pcl::Vertices> faces;
    for (std::size_t i = 0; i < textureMesh->tex_polygons.size(); ++i) {
        faces.insert(faces.end(), textureMesh->tex_polygons[i].begin(),
                     textureMesh->tex_polygons[i].end());
    }
    ccMesh* newMesh = Convert(textureMesh->cloud, faces);
    if (!newMesh) {
        return nullptr;
    }
    QString name("texture-mesh");
    newMesh->setName(name);

    // create texture coordinates
    TextureCoordsContainer* texCoords = new TextureCoordsContainer();
    if (texCoords) {
        texCoords->reserve(nr_faces);

        bool allocated = texCoords->isAllocated();

        allocated &= newMesh->reservePerTriangleTexCoordIndexes();
        allocated &= newMesh->reservePerTriangleMtlIndexes();

        if (!allocated) {
            delete texCoords;
            CVLog::Warning(QStringLiteral("[pcl2cc::Convert] Cannot allocate "
                                          "texture coordinates for mesh '%1'")
                                   .arg(name));
        } else {
            newMesh->setTexCoordinatesTable(texCoords);
        }

        for (std::size_t m = 0; m < nr_meshes; ++m) {
            if (textureMesh->tex_coordinates.empty()) continue;
            for (const auto& coordinate : textureMesh->tex_coordinates[m]) {
                const TexCoords2D coord{coordinate[0], coordinate[1]};
                texCoords->addElement(coord);
            }
        }
    }

    // materials and texture file
    ccMaterialSet* materialSet = new ccMaterialSet("Materials");
    if (!textureMesh->tex_materials.empty()) {
        for (std::size_t m = 0; m < nr_meshes; ++m) {
            auto newMaterial = ccMaterial::Shared(new ccMaterial());
            FromPCLMaterial(textureMesh->tex_materials[m], newMaterial);
            materialSet->addMaterial(newMaterial);

            for (std::size_t i = 0; i < textureMesh->tex_polygons[m].size();
                 ++i) {
                CCVector3i texCoordIndexes;
                for (std::size_t j = 0;
                     j < textureMesh->tex_polygons[m][i].vertices.size(); ++j) {
                    texCoordIndexes.u[j] = static_cast<int>(
                            textureMesh->tex_polygons[m][i].vertices[j]);
                }

                // texture coordinates
                newMesh->addTriangleMtlIndex(0);
                newMesh->addTriangleTexCoordIndexes(texCoordIndexes.x,
                                                    texCoordIndexes.y,
                                                    texCoordIndexes.z);
            }
        }
    }

    if (materialSet != nullptr) {
        newMesh->setMaterialSet(materialSet);
        newMesh->showMaterials(true);
    }

    if (newMesh->size() == 0) {
        CVLog::Warning(
                QStringLiteral(
                        "[pcl2cc::Convert] Mesh '%1' does not have any faces")
                        .arg(name));
        delete newMesh;
        return nullptr;
    }

    if (!newMesh->getAssociatedCloud()->hasNormals()) {
        CVLog::Warning(
                QStringLiteral("[pcl2cc::Convert] Mesh '%1' does not have "
                               "normals - will compute them per vertex")
                        .arg(name));
        newMesh->computeNormals(true);
    }

    newMesh->showNormals(true);
    newMesh->showColors(newMesh->hasColors());
    return newMesh;
}

ccPointCloud* pcl2cc::Convert(const PCLCloud& pclCloud,
                              bool ignoreScalars /* = false*/,
                              bool ignoreRgb /* = false*/) {
    // retrieve the valid fields
    std::list<std::string> fields;
    uint8_t coordinateType = 0;
    for (const auto& field : pclCloud.fields) {
        if (field.name != "_")  // PCL padding fields
        {
            fields.push_back(field.name);
        }

        if (coordinateType == 0 &&
            (field.name == "x" || field.name == "y" || field.name == "z")) {
            coordinateType = field.datatype;
        }
    }

    // begin with checks and conversions
    // be sure we have x, y, and z fields
    if (!ExistField(pclCloud, "x") || !ExistField(pclCloud, "y") ||
        !ExistField(pclCloud, "z")) {
        return nullptr;
    }

    // create cloud
    ccPointCloud* ccCloud = new ccPointCloud();
    size_t expectedPointCount = GetNumberOfPoints(pclCloud);
    if (expectedPointCount != 0) {
        // push points inside
        if (!CopyXYZ(pclCloud, *ccCloud, coordinateType)) {
            delete ccCloud;
            return nullptr;
        }
    }

    // remove x,y,z fields from the vector of field names
    fields.remove("x");
    fields.remove("y");
    fields.remove("z");

    // do we have normals?
    if (ExistField(pclCloud, "normal_x") || ExistField(pclCloud, "normal_y") ||
        ExistField(pclCloud, "normal_z")) {
        CopyNormals(pclCloud, *ccCloud);

        // remove the corresponding fields
        fields.remove("normal_x");
        fields.remove("normal_y");
        fields.remove("normal_z");
    }

    // do we have colors?
    if (!ignoreRgb) {
        // The same for colors
        if (ExistField(pclCloud, "rgb")) {
            CopyRGB(pclCloud, *ccCloud);

            // remove the corresponding field
            fields.remove("rgb");
        }
        // The same for colors
        else if (ExistField(pclCloud, "rgba")) {
            CopyRGB(pclCloud, *ccCloud);

            // remove the corresponding field
            fields.remove("rgba");
        }
    }

    // All the remaining fields will be stored as scalar fields
    if (!ignoreScalars) {
        for (const std::string& name : fields) {
            CopyScalarField(pclCloud, name, *ccCloud);
        }
    }

    return ccCloud;
}

ccMesh* pcl2cc::Convert(const PCLCloud& pclCloud,
                        const std::vector<pcl::Vertices>& polygons,
                        bool ignoreScalars /* = false*/,
                        bool ignoreRgb /* = false*/) {
    ccPointCloud* vertices = Convert(pclCloud, ignoreScalars, ignoreRgb);
    if (!vertices) return nullptr;
    vertices->setName("vertices");
    // vertices->showNormals(false);

    // mesh
    ccMesh* mesh = new ccMesh(vertices);
    mesh->setName("Mesh");

    size_t triNum = polygons.size();

    if (!mesh->reserve(triNum)) {
        assert(false);
        return nullptr;
    }
    for (size_t i = 0; i < triNum; ++i) {
        const pcl::Vertices& tri = polygons[i];
        mesh->addTriangle(tri.vertices[0], tri.vertices[1], tri.vertices[2]);
    }

    // do some cleaning
    {
        vertices->shrinkToFit();
        mesh->shrinkToFit();
        NormsIndexesTableType* normals = mesh->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    vertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    vertices->setLocked(false);
    mesh->addChild(vertices);

    return mesh;
}
