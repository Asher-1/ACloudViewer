// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file Vtk2Cc.cpp
 * @brief Implementation of VTK to CloudViewer data conversion.
 */

#include <Converters/Vtk2Cc.h>

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVLog.h>

// CV_DB_LIB
#include <ecvGenericMesh.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMaterial.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvNormalVectors.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

// VTK
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkFieldData.h>
#include <vtkFloatArray.h>
#include <vtkIdTypeArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkStringArray.h>
#include <vtkUnsignedCharArray.h>

// System
#include <cstring>
#include <vector>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

namespace Converters {

namespace {

bool ReadHasSourceRGBFlag(vtkFieldData* fieldData) {
    if (!fieldData) {
        return false;
    }

    vtkAbstractArray* array = fieldData->GetAbstractArray("HasSourceRGB");
    if (!array || array->GetNumberOfTuples() < 1) {
        return false;
    }

    if (auto* intArray = vtkIntArray::SafeDownCast(array)) {
        return intArray->GetValue(0) == 1;
    }

    if (auto* stringArray = vtkStringArray::SafeDownCast(array)) {
        const std::string value = stringArray->GetValue(0);
        return value == "1" || value == "true" || value == "True";
    }

    return false;
}

bool IsReservedPointArrayName(const char* arrayName) {
    if (!arrayName || arrayName[0] == '\0') {
        return true;
    }

    if (strcmp(arrayName, "Normals") == 0 ||
        strcmp(arrayName, "SourceRGB") == 0 || strcmp(arrayName, "RGB") == 0 ||
        strcmp(arrayName, "Colors") == 0 || strcmp(arrayName, "rgba") == 0 ||
        strcmp(arrayName, "rgb") == 0) {
        return true;
    }

    return strncmp(arrayName, "TCoords", 7) == 0;
}

bool IsColorArrayName(const char* arrayName) {
    if (!arrayName) {
        return false;
    }
    return strcmp(arrayName, "RGB") == 0 || strcmp(arrayName, "Colors") == 0 ||
           strcmp(arrayName, "rgba") == 0 || strcmp(arrayName, "rgb") == 0;
}

vtkUnsignedCharArray* FindColorArray(vtkPointData* pointData,
                                     bool hasSourceRGBFlag) {
    if (!pointData) {
        return nullptr;
    }

    if (hasSourceRGBFlag || pointData->GetArray("SourceRGB")) {
        if (auto* sourceRGB = vtkUnsignedCharArray::SafeDownCast(
                    pointData->GetArray("SourceRGB"))) {
            if (sourceRGB->GetNumberOfComponents() >= 3) {
                return sourceRGB;
            }
        }
    }

    if (vtkDataArray* scalars = pointData->GetScalars()) {
        if (scalars->GetNumberOfComponents() >= 3) {
            if (auto* colors = vtkUnsignedCharArray::SafeDownCast(scalars)) {
                const char* name = scalars->GetName();
                if (hasSourceRGBFlag || !name || IsColorArrayName(name)) {
                    return colors;
                }
            }
        }
    }

    const char* colorArrayNames[] = {"RGB", "Colors", "rgba", "rgb"};
    for (const char* name : colorArrayNames) {
        vtkDataArray* arr = pointData->GetArray(name);
        if (!arr || arr->GetNumberOfComponents() < 3) {
            continue;
        }
        if (auto* colors = vtkUnsignedCharArray::SafeDownCast(arr)) {
            return colors;
        }
    }

    return nullptr;
}

QString ReadDatasetName(vtkFieldData* fieldData) {
    if (!fieldData) {
        return {};
    }

    auto* datasetName = vtkStringArray::SafeDownCast(
            fieldData->GetAbstractArray("DatasetName"));
    if (!datasetName || datasetName->GetNumberOfTuples() < 1) {
        return {};
    }

    return QString::fromStdString(datasetName->GetValue(0));
}

QString ReadActiveScalarFieldName(vtkFieldData* fieldData) {
    if (!fieldData) {
        return {};
    }

    auto* activeSf = vtkStringArray::SafeDownCast(
            fieldData->GetAbstractArray("ActiveScalarField"));
    if (!activeSf || activeSf->GetNumberOfTuples() < 1) {
        return {};
    }

    return QString::fromStdString(activeSf->GetValue(0));
}

const ccPointCloud* GetSourcePointCloud(const ccHObject* sourceEntity) {
    if (!sourceEntity) {
        return nullptr;
    }

    if (sourceEntity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return ccHObjectCaster::ToPointCloud(
                const_cast<ccHObject*>(sourceEntity));
    }

    if (sourceEntity->isKindOf(CV_TYPES::MESH)) {
        ccMesh* mesh =
                ccHObjectCaster::ToMesh(const_cast<ccHObject*>(sourceEntity));
        return mesh ? ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud())
                    : nullptr;
    }

    return nullptr;
}

const ccMesh* GetSourceMesh(const ccHObject* sourceEntity) {
    if (!sourceEntity || !sourceEntity->isKindOf(CV_TYPES::MESH)) {
        return nullptr;
    }
    return ccHObjectCaster::ToMesh(const_cast<ccHObject*>(sourceEntity));
}

void ApplyActiveScalarField(ccPointCloud* cloud,
                            vtkFieldData* fieldData,
                            const ccPointCloud* sourceCloud) {
    if (!cloud || !cloud->hasScalarFields()) {
        return;
    }

    QString activeName = ReadActiveScalarFieldName(fieldData);
    if (activeName.isEmpty() && sourceCloud &&
        sourceCloud->getCurrentDisplayedScalarField()) {
        activeName = QString::fromStdString(
                sourceCloud->getCurrentDisplayedScalarField()->getName());
    }

    if (activeName.isEmpty()) {
        return;
    }

    const int sfIdx =
            cloud->getScalarFieldIndexByName(activeName.toUtf8().constData());
    if (sfIdx < 0) {
        return;
    }

    cloud->setCurrentDisplayedScalarField(sfIdx);
    if (sourceCloud) {
        const int srcIdx = sourceCloud->getScalarFieldIndexByName(
                activeName.toUtf8().constData());
        if (srcIdx >= 0) {
            ccScalarField* outSf =
                    static_cast<ccScalarField*>(cloud->getScalarField(sfIdx));
            const ccScalarField* srcSf = static_cast<const ccScalarField*>(
                    sourceCloud->getScalarField(srcIdx));
            if (outSf && srcSf) {
                outSf->importParametersFrom(srcSf);
            }
        }
    }
    cloud->showSF(true);
}

void ImportScalarFieldParameters(ccPointCloud* cloud,
                                 const ccPointCloud* sourceCloud) {
    if (!cloud || !sourceCloud) {
        return;
    }

    for (unsigned i = 0; i < cloud->getNumberOfScalarFields(); ++i) {
        ccScalarField* outSf =
                static_cast<ccScalarField*>(cloud->getScalarField(i));
        if (!outSf) {
            continue;
        }
        const int srcIdx =
                sourceCloud->getScalarFieldIndexByName(outSf->getName());
        if (srcIdx >= 0) {
            const ccScalarField* srcSf = static_cast<const ccScalarField*>(
                    sourceCloud->getScalarField(srcIdx));
            if (srcSf) {
                outSf->importParametersFrom(srcSf);
            }
        }
    }
}

void ApplySourceCloudAttributes(ccPointCloud* cloud,
                                const ccPointCloud* sourceCloud) {
    if (!cloud || !sourceCloud) {
        return;
    }

    cloud->importParametersFrom(sourceCloud);
    cloud->showNormals(sourceCloud->normalsShown());
    cloud->showColors(sourceCloud->colorsShown());
    cloud->showSF(sourceCloud->sfShown());
    ImportScalarFieldParameters(cloud, sourceCloud);
    ApplyActiveScalarField(cloud, nullptr, sourceCloud);
}

void ApplySourceMeshAttributes(ccMesh* mesh, const ccMesh* sourceMesh) {
    if (!mesh || !sourceMesh) {
        return;
    }

    mesh->importParametersFrom(sourceMesh);
    mesh->showNormals(sourceMesh->normalsShown());
    mesh->showColors(sourceMesh->colorsShown());
    mesh->showSF(sourceMesh->sfShown());
    mesh->showMaterials(sourceMesh->materialsShown());

    ccPointCloud* outVertices =
            ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
    ccPointCloud* srcVertices =
            ccHObjectCaster::ToPointCloud(sourceMesh->getAssociatedCloud());
    if (outVertices && srcVertices) {
        ApplySourceCloudAttributes(outVertices, srcVertices);
    }
}

bool IsValidTextureCoordinate(float u, float v) {
    // Cc2Vtk marks non-applicable per-material coords as (-1, -1).
    // After vtkClipPolyData interpolation, these sentinels can become values
    // like (-0.5, -0.3) which are still invalid.
    if (u <= -0.5f && v <= -0.5f) {
        return false;
    }
    return u >= 0.0f && v >= 0.0f;
}

vtkFloatArray* FindUnifiedTCoordsArray(vtkPointData* pointData) {
    if (!pointData) return nullptr;
    // The unified array is named exactly "TCoords" (not "TCoords0", etc.)
    vtkDataArray* arr = pointData->GetArray("TCoords");
    if (arr && arr->GetNumberOfComponents() >= 2 &&
        arr->GetNumberOfTuples() > 0) {
        return vtkFloatArray::SafeDownCast(arr);
    }
    return nullptr;
}

void ReadTextureCoordinate(vtkFloatArray* tcoords,
                           vtkIdType pointIndex,
                           TexCoords2D& coord) {
    coord = TexCoords2D(0.0f, 0.0f);
    if (!tcoords || pointIndex < 0 ||
        pointIndex >= tcoords->GetNumberOfTuples()) {
        return;
    }

    float uv[2];
    tcoords->GetTypedTuple(pointIndex, uv);
    coord = TexCoords2D(uv[0], uv[1]);
}

std::vector<vtkFloatArray*> CollectTextureCoordinateArrays(
        vtkPointData* pointData) {
    std::vector<vtkFloatArray*> arrays;
    if (!pointData) {
        return arrays;
    }

    auto addArray = [&arrays](vtkFloatArray* tcoords) {
        if (!tcoords) {
            return;
        }
        for (vtkFloatArray* existing : arrays) {
            if (existing == tcoords) {
                return;
            }
        }
        arrays.push_back(tcoords);
    };

    if (vtkDataArray* activeTCoords = pointData->GetTCoords()) {
        addArray(vtkFloatArray::SafeDownCast(activeTCoords));
    }

    for (int i = 0; i < pointData->GetNumberOfArrays(); ++i) {
        vtkDataArray* array = pointData->GetArray(i);
        if (!array || array->GetNumberOfComponents() < 2) {
            continue;
        }

        const char* name = array->GetName();
        if (!name || strncmp(name, "TCoords", 7) != 0) {
            continue;
        }

        addArray(vtkFloatArray::SafeDownCast(array));
    }

    return arrays;
}

vtkFloatArray* SelectTextureCoordinateArray(
        const std::vector<vtkFloatArray*>& materialTCoords, int materialIndex) {
    if (materialTCoords.empty()) {
        return nullptr;
    }
    if (materialIndex >= 0 &&
        materialIndex < static_cast<int>(materialTCoords.size())) {
        return materialTCoords[static_cast<size_t>(materialIndex)];
    }
    return materialTCoords.front();
}

int ReadTriangleMaterialIndex(vtkPolyData* polydata, vtkIdType cellId) {
    if (!polydata) {
        return -1;
    }

    vtkCellData* cellData = polydata->GetCellData();
    if (!cellData) {
        return -1;
    }

    vtkIntArray* mtlIndices =
            vtkIntArray::SafeDownCast(cellData->GetArray("MaterialIndex"));
    if (!mtlIndices || cellId < 0 ||
        cellId >= mtlIndices->GetNumberOfTuples()) {
        return -1;
    }

    return mtlIndices->GetValue(cellId);
}

int FindMaterialIndexForTriangle(
        const std::vector<vtkFloatArray*>& materialTCoords,
        vtkIdType i0,
        vtkIdType i1,
        vtkIdType i2) {
    if (materialTCoords.empty()) {
        return -1;
    }

    if (materialTCoords.size() == 1) {
        TexCoords2D tc0, tc1, tc2;
        ReadTextureCoordinate(materialTCoords[0], i0, tc0);
        ReadTextureCoordinate(materialTCoords[0], i1, tc1);
        ReadTextureCoordinate(materialTCoords[0], i2, tc2);
        if (IsValidTextureCoordinate(tc0.tx, tc0.ty) ||
            IsValidTextureCoordinate(tc1.tx, tc1.ty) ||
            IsValidTextureCoordinate(tc2.tx, tc2.ty)) {
            return 0;
        }
        return -1;
    }

    for (size_t matIdx = 0; matIdx < materialTCoords.size(); ++matIdx) {
        TexCoords2D tc0, tc1, tc2;
        ReadTextureCoordinate(materialTCoords[matIdx], i0, tc0);
        ReadTextureCoordinate(materialTCoords[matIdx], i1, tc1);
        ReadTextureCoordinate(materialTCoords[matIdx], i2, tc2);

        if (IsValidTextureCoordinate(tc0.tx, tc0.ty) &&
            IsValidTextureCoordinate(tc1.tx, tc1.ty) &&
            IsValidTextureCoordinate(tc2.tx, tc2.ty)) {
            return static_cast<int>(matIdx);
        }
    }

    return -1;
}

int AppendCompressedNormal(NormsIndexesTableType* normsTable,
                           const float normal[3]) {
    if (!normsTable) {
        return -1;
    }

    CCVector3 N(static_cast<PointCoordinateType>(normal[0]),
                static_cast<PointCoordinateType>(normal[1]),
                static_cast<PointCoordinateType>(normal[2]));
    CompressedNormType compressed = ccNormalVectors::GetNormIndex(N);

    for (unsigned i = 0; i < normsTable->size(); ++i) {
        if (normsTable->at(i) == compressed) {
            return static_cast<int>(i);
        }
    }

    normsTable->push_back(compressed);
    return static_cast<int>(normsTable->size() - 1);
}

void RestoreMeshTriangleNormals(ccMesh* mesh,
                                vtkPolyData* polydata,
                                const std::vector<Tuple3i>& triangleIndices,
                                const std::vector<vtkIdType>& triangleCellIds,
                                bool silent) {
    if (!mesh || !polydata || triangleIndices.empty()) {
        return;
    }

    vtkCellData* cellData = polydata->GetCellData();
    vtkFloatArray* cellNormals =
            cellData ? vtkFloatArray::SafeDownCast(cellData->GetNormals())
                     : nullptr;
    if (!cellNormals) {
        cellNormals = cellData ? vtkFloatArray::SafeDownCast(
                                         cellData->GetArray("Normals"))
                               : nullptr;
    }

    if (cellNormals && cellNormals->GetNumberOfComponents() == 3) {
        NormsIndexesTableType* normsTable = new NormsIndexesTableType();
        if (!mesh->reservePerTriangleNormalIndexes()) {
            normsTable->release();
            if (!silent) {
                CVLog::Warning(
                        "[Vtk2Cc::ConvertToMesh] not enough memory for "
                        "triangle normal indexes!");
            }
            return;
        }

        mesh->setTriNormsTable(normsTable);
        for (size_t triIdx = 0; triIdx < triangleIndices.size(); ++triIdx) {
            const vtkIdType cellId = (triIdx < triangleCellIds.size())
                                             ? triangleCellIds[triIdx]
                                             : static_cast<vtkIdType>(triIdx);
            if (cellId >= cellNormals->GetNumberOfTuples()) continue;
            float normal[3];
            cellNormals->GetTypedTuple(cellId, normal);
            const int normalIndex = AppendCompressedNormal(normsTable, normal);
            mesh->addTriangleNormalIndexes(normalIndex, normalIndex,
                                           normalIndex);
        }
        mesh->showNormals(true);
        return;
    }

    vtkPointData* pointData = polydata->GetPointData();
    vtkFloatArray* pointNormals =
            pointData ? vtkFloatArray::SafeDownCast(pointData->GetNormals())
                      : nullptr;
    if (!pointNormals) {
        pointNormals = pointData ? vtkFloatArray::SafeDownCast(
                                           pointData->GetArray("Normals"))
                                 : nullptr;
    }

    if (!pointNormals || pointNormals->GetNumberOfComponents() != 3) {
        return;
    }

    NormsIndexesTableType* normsTable = new NormsIndexesTableType();
    if (!mesh->reservePerTriangleNormalIndexes()) {
        normsTable->release();
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToMesh] not enough memory for triangle "
                    "normal indexes!");
        }
        return;
    }

    mesh->setTriNormsTable(normsTable);
    for (const Tuple3i& tri : triangleIndices) {
        float n0[3], n1[3], n2[3];
        pointNormals->GetTypedTuple(tri.u[0], n0);
        pointNormals->GetTypedTuple(tri.u[1], n1);
        pointNormals->GetTypedTuple(tri.u[2], n2);

        const int i0 = AppendCompressedNormal(normsTable, n0);
        const int i1 = AppendCompressedNormal(normsTable, n1);
        const int i2 = AppendCompressedNormal(normsTable, n2);
        mesh->addTriangleNormalIndexes(i0, i1, i2);
    }
    mesh->showNormals(true);
}

void RestoreMeshTextureCoordinates(
        ccMesh* mesh,
        vtkPolyData* polydata,
        const std::vector<Tuple3i>& triangleIndices,
        const std::vector<vtkIdType>& triangleCellIds,
        const ccMesh* sourceMesh,
        bool silent) {
    if (!mesh || !polydata || triangleIndices.empty()) {
        return;
    }

    // Prefer the unified "TCoords" array: it holds correct per-vertex UVs for
    // ALL materials and is properly interpolated by vtkClipPolyData at clip
    // boundaries. Per-material arrays ("TCoords0", "TCoords1"...) use (-1,-1)
    // sentinels for non-applicable vertices which get corrupted by
    // interpolation.
    vtkFloatArray* unifiedTCoords =
            FindUnifiedTCoordsArray(polydata->GetPointData());

    std::vector<vtkFloatArray*> materialTCoords;
    if (!unifiedTCoords) {
        materialTCoords =
                CollectTextureCoordinateArrays(polydata->GetPointData());
        if (materialTCoords.empty()) {
            return;
        }
    }

    auto* texTable = new TextureCoordsContainer();
    texTable->reserve(triangleIndices.size() * 3);
    if (!texTable->isAllocated()) {
        texTable->release();
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToMesh] not enough memory for texture "
                    "coordinates!");
        }
        return;
    }

    mesh->setTexCoordinatesTable(texTable);
    if (!mesh->reservePerTriangleTexCoordIndexes()) {
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToMesh] not enough memory for texture "
                    "coordinate indexes!");
        }
        return;
    }

    for (size_t triIdx = 0; triIdx < triangleIndices.size(); ++triIdx) {
        const Tuple3i& tri = triangleIndices[triIdx];

        vtkFloatArray* tcoordsArray = unifiedTCoords;
        if (!tcoordsArray) {
            const vtkIdType cellId = (triIdx < triangleCellIds.size())
                                             ? triangleCellIds[triIdx]
                                             : static_cast<vtkIdType>(triIdx);
            int mtlIndex = ReadTriangleMaterialIndex(polydata, cellId);
            if (mtlIndex < 0) {
                mtlIndex = FindMaterialIndexForTriangle(
                        materialTCoords, tri.u[0], tri.u[1], tri.u[2]);
            }
            if (mtlIndex < 0 && sourceMesh && sourceMesh->hasMaterials()) {
                mtlIndex = 0;
            }
            tcoordsArray =
                    SelectTextureCoordinateArray(materialTCoords, mtlIndex);
        }
        TexCoords2D tc0, tc1, tc2;
        if (tcoordsArray) {
            ReadTextureCoordinate(tcoordsArray, tri.u[0], tc0);
            ReadTextureCoordinate(tcoordsArray, tri.u[1], tc1);
            ReadTextureCoordinate(tcoordsArray, tri.u[2], tc2);
        }

        const int i0 = static_cast<int>(texTable->size());
        texTable->addElement(tc0);
        const int i1 = static_cast<int>(texTable->size());
        texTable->addElement(tc1);
        const int i2 = static_cast<int>(texTable->size());
        texTable->addElement(tc2);
        mesh->addTriangleTexCoordIndexes(i0, i1, i2);
    }
}

void RestoreMeshMaterials(ccMesh* mesh,
                          vtkPolyData* polydata,
                          const std::vector<Tuple3i>& triangleIndices,
                          const std::vector<vtkIdType>& triangleCellIds,
                          const ccMesh* sourceMesh,
                          bool silent) {
    if (!mesh || !polydata || triangleIndices.empty()) {
        return;
    }

    ccMaterialSet* materialSet = nullptr;
    if (sourceMesh && sourceMesh->hasMaterials()) {
        materialSet = sourceMesh->getMaterialSet()->clone();
    }

    vtkFieldData* fieldData = polydata->GetFieldData();
    if (!materialSet && fieldData) {
        auto* materialNames = vtkStringArray::SafeDownCast(
                fieldData->GetAbstractArray("MaterialNames"));
        if (materialNames && materialNames->GetNumberOfTuples() > 0) {
            materialSet = new ccMaterialSet("Materials");
            for (vtkIdType i = 0; i < materialNames->GetNumberOfTuples(); ++i) {
                const QString name =
                        QString::fromStdString(materialNames->GetValue(i));
                materialSet->addMaterial(
                        ccMaterial::Shared(new ccMaterial(name)));
            }
        }
    }

    if (!materialSet || materialSet->size() == 0) {
        if (materialSet) {
            materialSet->release();
        }
        return;
    }

    mesh->setMaterialSet(materialSet);

    if (sourceMesh) {
        const QVariant mtlData = sourceMesh->getMetaData("MTL_FILENAME");
        if (mtlData.isValid() && !mtlData.toString().isEmpty()) {
            mesh->setMetaData("MTL_FILENAME", mtlData);
        }
    } else if (fieldData) {
        auto* materialLibraries = vtkStringArray::SafeDownCast(
                fieldData->GetAbstractArray("MaterialLibraries"));
        if (materialLibraries && materialLibraries->GetNumberOfTuples() > 0) {
            mesh->setMetaData(
                    "MTL_FILENAME",
                    QString::fromStdString(materialLibraries->GetValue(0)));
        }
    }

    if (!mesh->reservePerTriangleMtlIndexes()) {
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToMesh] not enough memory for material "
                    "indexes!");
        }
        return;
    }

    const std::vector<vtkFloatArray*> materialTCoords =
            CollectTextureCoordinateArrays(polydata->GetPointData());
    for (size_t triIdx = 0; triIdx < triangleIndices.size(); ++triIdx) {
        const Tuple3i& tri = triangleIndices[triIdx];
        const vtkIdType cellId = (triIdx < triangleCellIds.size())
                                         ? triangleCellIds[triIdx]
                                         : static_cast<vtkIdType>(triIdx);
        int mtlIndex = ReadTriangleMaterialIndex(polydata, cellId);
        if (mtlIndex < 0) {
            mtlIndex = FindMaterialIndexForTriangle(materialTCoords, tri.u[0],
                                                    tri.u[1], tri.u[2]);
        }
        if (mtlIndex < 0 && materialSet->size() > 0) {
            mtlIndex = 0;
        }
        mesh->addTriangleMtlIndex(mtlIndex);
    }

    mesh->showMaterials(true);
}

}  // namespace

ccHObject* Vtk2Cc::Convert(vtkPolyData* polydata,
                           bool asMesh,
                           const Vtk2CcOptions& options) {
    if (asMesh) {
        return ConvertToMesh(polydata, options);
    }
    return ConvertToPointCloud(polydata, options);
}

ccPointCloud* Vtk2Cc::ConvertToPointCloud(vtkPolyData* polydata, bool silent) {
    Vtk2CcOptions options;
    options.silent = silent;
    return ConvertToPointCloud(polydata, options);
}

ccPointCloud* Vtk2Cc::ConvertToPointCloud(vtkPolyData* polydata,
                                          const Vtk2CcOptions& options) {
    const bool silent = options.silent;
    if (!polydata) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToPointCloud] polydata is nullptr");
        }
        return nullptr;
    }

    vtkIdType pointCount = polydata->GetNumberOfPoints();
    if (pointCount == 0) {
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToPointCloud] polydata has 0 points");
        }
        return nullptr;
    }

    vtkPointData* pointData = polydata->GetPointData();
    vtkFieldData* fieldData = polydata->GetFieldData();

    const bool hasSourceRGBFlag = ReadHasSourceRGBFlag(fieldData);
    vtkUnsignedCharArray* colors = FindColorArray(pointData, hasSourceRGBFlag);

    vtkFloatArray* normals = nullptr;
    if (pointData) {
        normals = vtkFloatArray::SafeDownCast(pointData->GetNormals());
        if (!normals) {
            vtkDataArray* arr = pointData->GetArray("Normals");
            if (arr && arr->GetNumberOfComponents() == 3) {
                normals = vtkFloatArray::SafeDownCast(arr);
            }
        }
    }

    const QString datasetName = ReadDatasetName(fieldData);
    const QString entityName = !options.nameOverride.isEmpty()
                                       ? options.nameOverride
                                       : datasetName;

    ccPointCloud* cloud =
            new ccPointCloud(entityName.isEmpty() ? "vertices" : entityName);

    if (!cloud->resize(static_cast<unsigned>(pointCount))) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToPointCloud] not enough memory!");
        }
        delete cloud;
        return nullptr;
    }

    if (normals && !cloud->reserveTheNormsTable()) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc] not enough memory for normals!");
        }
        delete cloud;
        return nullptr;
    }

    if (colors && !cloud->reserveTheRGBTable()) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc] not enough memory for colors!");
        }
        delete cloud;
        return nullptr;
    }

    for (vtkIdType i = 0; i < pointCount; ++i) {
        double coordinate[3];
        polydata->GetPoint(i, coordinate);
        cloud->setPoint(static_cast<std::size_t>(i),
                        CCVector3::fromArray(coordinate));
        if (normals) {
            float normal[3];
            normals->GetTupleValue(i, normal);
            CCVector3 N(static_cast<PointCoordinateType>(normal[0]),
                        static_cast<PointCoordinateType>(normal[1]),
                        static_cast<PointCoordinateType>(normal[2]));
            cloud->addNorm(N);
        }
        if (colors) {
            unsigned char color[4];
            colors->GetTupleValue(i, color);
            ecvColor::Rgb C(static_cast<ColorCompType>(color[0]),
                            static_cast<ColorCompType>(color[1]),
                            static_cast<ColorCompType>(color[2]));
            cloud->addRGBColor(C);
        }
    }

    if (normals) {
        cloud->showNormals(true);
    }
    if (colors) {
        cloud->showColors(true);
    }

    if (pointData && pointCount > 0) {
        int numArrays = pointData->GetNumberOfArrays();
        for (int i = 0; i < numArrays; ++i) {
            vtkDataArray* dataArray = pointData->GetArray(i);
            if (!dataArray || dataArray->GetNumberOfComponents() != 1) {
                continue;
            }
            if (dataArray->GetNumberOfTuples() < pointCount) {
                continue;
            }
            if (dataArray == reinterpret_cast<vtkDataArray*>(colors) ||
                dataArray == reinterpret_cast<vtkDataArray*>(normals)) {
                continue;
            }

            const char* arrayName = dataArray->GetName();
            if (IsReservedPointArrayName(arrayName)) {
                continue;
            }

            ccScalarField* scalarField = new ccScalarField(arrayName);
            if (!scalarField->resizeSafe(static_cast<unsigned>(pointCount))) {
                scalarField->release();
                continue;
            }

            for (vtkIdType j = 0; j < pointCount; ++j) {
                double value = dataArray->GetTuple1(j);
                scalarField->setValue(static_cast<unsigned>(j),
                                      static_cast<ScalarType>(value));
            }

            scalarField->computeMinAndMax();
            int sfIdx = cloud->addScalarField(scalarField);
            if (sfIdx < 0) {
                scalarField->release();
            } else {
                QString fieldName = QString(arrayName).toLower();
                if (fieldName.contains("label") ||
                    fieldName.contains("class") ||
                    fieldName.contains("segment") ||
                    fieldName.contains("cluster")) {
                    cloud->setCurrentDisplayedScalarField(sfIdx);
                    cloud->showSF(true);
                }
            }
        }
    }

    const ccPointCloud* sourceCloud = GetSourcePointCloud(options.sourceEntity);
    ApplyActiveScalarField(cloud, fieldData, sourceCloud);
    if (sourceCloud) {
        ApplySourceCloudAttributes(cloud, sourceCloud);
    }

    return cloud;
}

ccMesh* Vtk2Cc::ConvertToMesh(vtkPolyData* polydata, bool silent) {
    Vtk2CcOptions options;
    options.silent = silent;
    return ConvertToMesh(polydata, options);
}

ccMesh* Vtk2Cc::ConvertToMesh(vtkPolyData* polydata,
                              const Vtk2CcOptions& options) {
    const bool silent = options.silent;
    if (!polydata) {
        return nullptr;
    }

    vtkSmartPointer<vtkPoints> mesh_points = polydata->GetPoints();
    if (!mesh_points) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToMesh] polydata has no points!");
        }
        return nullptr;
    }

    unsigned nr_points =
            static_cast<unsigned>(mesh_points->GetNumberOfPoints());
    unsigned nr_polygons = static_cast<unsigned>(polydata->GetNumberOfPolys());
    if (nr_points == 0) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToMesh] cannot find points data!");
        }
        return nullptr;
    }

    Vtk2CcOptions vertexOptions = options;
    vertexOptions.nameOverride.clear();

    ccPointCloud* vertices = ConvertToPointCloud(polydata, vertexOptions);
    if (!vertices) {
        return nullptr;
    }
    vertices->setEnabled(false);
    vertices->setLocked(false);

    ccMesh* mesh = new ccMesh(vertices);
    const QString datasetName = ReadDatasetName(polydata->GetFieldData());
    const QString meshName =
            !options.nameOverride.isEmpty()
                    ? options.nameOverride
                    : (datasetName.isEmpty() ? "Mesh" : datasetName);
    mesh->setName(meshName);
    mesh->addChild(vertices);

    if (!mesh->reserve(nr_polygons)) {
        if (!silent) {
            CVLog::Warning("[Vtk2Cc::ConvertToMesh] not enough memory!");
        }
        delete mesh;
        return nullptr;
    }

#ifdef VTK_CELL_ARRAY_V2
    const vtkIdType* cell_points;
#else
    vtkIdType* cell_points;
#endif
    vtkIdType nr_cell_points;
    vtkCellArray* mesh_polygons = polydata->GetPolys();
    mesh_polygons->InitTraversal();
    unsigned int validTriangles = 0;
    unsigned int skippedCells = 0;
    std::vector<Tuple3i> triangleIndices;
    triangleIndices.reserve(nr_polygons);

    // Track the actual cell index for each kept triangle so that cell data
    // (MaterialIndex) can be read at the correct offset. Cell data is indexed
    // by global cell ID = nVerts + nLines + polyIndex.
    const vtkIdType cellIdOffset =
            polydata->GetNumberOfVerts() + polydata->GetNumberOfLines();
    vtkIdType polyIndex = 0;
    std::vector<vtkIdType> triangleCellIds;
    triangleCellIds.reserve(nr_polygons);

    while (mesh_polygons->GetNextCell(nr_cell_points, cell_points)) {
        if (nr_cell_points != 3) {
            ++skippedCells;
            ++polyIndex;
            continue;
        }
        const Tuple3i tri(static_cast<int>(cell_points[0]),
                          static_cast<int>(cell_points[1]),
                          static_cast<int>(cell_points[2]));
        mesh->addTriangle(static_cast<unsigned>(tri.u[0]),
                          static_cast<unsigned>(tri.u[1]),
                          static_cast<unsigned>(tri.u[2]));
        triangleIndices.push_back(tri);
        triangleCellIds.push_back(cellIdOffset + polyIndex);
        ++validTriangles;
        ++polyIndex;
    }

    if (validTriangles == 0) {
        if (!silent) {
            CVLog::Warning(
                    "[Vtk2Cc::ConvertToMesh] No triangles found in polydata");
        }
        delete mesh;
        return nullptr;
    }

    if (skippedCells > 0 && !silent) {
        CVLog::Warning(
                QString("[Vtk2Cc::ConvertToMesh] Skipped %1 non-triangle "
                        "cell(s), added %2 triangle(s)")
                        .arg(skippedCells)
                        .arg(validTriangles));
    }

    vertices->shrinkToFit();
    mesh->shrinkToFit();
    NormsIndexesTableType* norms = mesh->getTriNormsTable();
    if (norms) {
        norms->shrink_to_fit();
    }

    const ccMesh* sourceMesh = GetSourceMesh(options.sourceEntity);

    RestoreMeshTriangleNormals(mesh, polydata, triangleIndices, triangleCellIds,
                               silent);
    RestoreMeshTextureCoordinates(mesh, polydata, triangleIndices,
                                  triangleCellIds, sourceMesh, silent);
    RestoreMeshMaterials(mesh, polydata, triangleIndices, triangleCellIds,
                         sourceMesh, silent);

    if (mesh->hasTextures()) {
        mesh->showMaterials(true);
    }

    if (sourceMesh) {
        ApplySourceMeshAttributes(mesh, sourceMesh);
    }

    return mesh;
}

ccPolyline* Vtk2Cc::ConvertToPolyline(vtkPolyData* polydata, bool silent) {
    if (!polydata) return nullptr;

    ccPointCloud* obj = ConvertToPointCloud(polydata, silent);
    if (!obj) {
        CVLog::Error(
                "[Vtk2Cc::ConvertToPolyline] failed to convert vtkPolyData "
                "to ccPointCloud");
        return nullptr;
    }

    if (obj->size() == 0) {
        CVLog::Warning(
                "[Vtk2Cc::ConvertToPolyline] polyline vertices is empty!");
        delete obj;
        return nullptr;
    }

    return ConvertToPolyline(obj);
}

ccPolyline* Vtk2Cc::ConvertToPolyline(ccPointCloud* vertices) {
    if (!vertices || !vertices->isKindOf(CV_TYPES::POINT_CLOUD)) {
        return nullptr;
    }

    ccPointCloud* polyVertices = ccHObjectCaster::ToPointCloud(vertices);
    if (!polyVertices) {
        return nullptr;
    }

    ccPolyline* curvePoly = new ccPolyline(polyVertices);
    if (!curvePoly) {
        return nullptr;
    }

    unsigned verticesCount = polyVertices->size();
    if (curvePoly->reserve(verticesCount)) {
        curvePoly->addPointIndex(0, verticesCount);
        curvePoly->setVisible(true);

        bool closed = false;
        CCVector3 start = CCVector3::fromArray(polyVertices->getPoint(0)->u);
        CCVector3 end = CCVector3::fromArray(
                polyVertices->getPoint(verticesCount - 1)->u);
        if (cloudViewer::LessThanEpsilon((end - start).norm())) {
            closed = true;
        }

        curvePoly->setClosed(closed);
        curvePoly->setName("polyline");
        curvePoly->addChild(polyVertices);
        curvePoly->showColors(true);
        curvePoly->setTempColor(ecvColor::green);
        curvePoly->set2DMode(false);
    } else {
        delete curvePoly;
        curvePoly = nullptr;
    }

    return curvePoly;
}

ccHObject::Container Vtk2Cc::ConvertToMultiPolylines(
        vtkPolyData* polydata, QString baseName, const ecvColor::Rgb& color) {
    ccHObject::Container container;

    vtkIdType iCells = polydata->GetNumberOfCells();
    for (vtkIdType i = 0; i < iCells; i++) {
        ccPointCloud* vertices = nullptr;
        vtkCell* cell = polydata->GetCell(i);
        vtkIdType ptsCount = cell->GetNumberOfPoints();
        if (ptsCount > 1) {
            vertices = new ccPointCloud("vertices");
            if (!vertices->reserve(static_cast<unsigned>(ptsCount))) {
                CVLog::Error("not enough memory to allocate vertices...");
                return container;
            }

            for (vtkIdType iPt = 0; iPt < ptsCount; ++iPt) {
                CCVector3 P =
                        CCVector3::fromArray(cell->GetPoints()->GetPoint(iPt));
                vertices->addPoint(P);
            }
        }

        if (vertices && vertices->size() == 0) {
            delete vertices;
            vertices = nullptr;
        }

        if (vertices) {
            vertices->setName("vertices");
            vertices->setEnabled(false);
            vertices->setPointSize(4);
            vertices->showColors(true);
            vertices->setTempColor(ecvColor::red);
            if (vertices->hasNormals()) vertices->showNormals(true);
            if (vertices->hasScalarFields()) {
                vertices->setCurrentDisplayedScalarField(0);
                vertices->showSF(true);
            }

            ccPolyline* poly = ConvertToPolyline(vertices);
            if (!poly) {
                delete vertices;
                continue;
            }

            QString contourName = baseName;
            if (poly->size() > 1) {
                contourName += QString(" (part %1)").arg(i + 1);
            }
            poly->setName(contourName);
            poly->showColors(true);
            poly->setTempColor(color);
            poly->set2DMode(false);

            container.push_back(poly);
        }
    }
    return container;
}

}  // namespace Converters
