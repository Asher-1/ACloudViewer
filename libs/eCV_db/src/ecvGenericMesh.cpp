//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "ecvGenericMesh.h"

// local
#include "ecvChunk.h"
#include "ecvColorScalesManager.h"
#include "ecvDisplayTools.h"
#include "ecvGenericPointCloud.h"
#include "ecvHObjectCaster.h"
#include "ecvMaterialSet.h"
#include "ecvNormalVectors.h"
#include "ecvPointCloud.h"
#include "ecvScalarField.h"

// cloudViewer
#include <CVPointCloud.h>
#include <GenericProgressCallback.h>
#include <GenericTriangle.h>
#include <MeshSamplingTools.h>
#include <ReferenceCloud.h>

// system
#include <QFileInfo>
#include <cassert>

ccGenericMesh::ccGenericMesh(QString name /*=QString()*/)
    : GenericIndexedMesh(),
      ccShiftedObject(name),
      m_triNormsShown(false),
      m_materialsShown(false),
      m_showWired(false),
      m_showPoints(false),
      m_stippling(false) {
    setVisible(true);
    lockVisibility(false);
}

void ccGenericMesh::showNormals(bool state) {
    showTriNorms(state);
    ccHObject::showNormals(state);
}

// stipple mask (for semi-transparent display of meshes)
static const GLubyte s_byte0 = 1 | 4 | 16 | 64;
static const GLubyte s_byte1 = 2 | 8 | 32 | 128;
static const GLubyte s_stippleMask[4 * 32] = {
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1,
        s_byte0, s_byte0, s_byte0, s_byte0, s_byte1, s_byte1, s_byte1, s_byte1};

// Vertex buffer
CCVector3* ccGenericMesh::GetVertexBuffer() {
    static CCVector3 s_xyzBuffer[ccChunk::SIZE * 3];
    return s_xyzBuffer;
}

// Normals buffer
CCVector3* ccGenericMesh::GetNormalsBuffer() {
    static CCVector3 s_normBuffer[ccChunk::SIZE * 3];
    return s_normBuffer;
}

// Colors buffer
ecvColor::Rgb* ccGenericMesh::GetColorsBuffer() {
    static ecvColor::Rgb s_rgbBuffer[ccChunk::SIZE * 3];
    return s_rgbBuffer;
}

// Vertex indexes buffer (for wired display)
static unsigned s_vertWireIndexes[ccChunk::SIZE * 6];
static bool s_vertIndexesInitialized = false;
unsigned* ccGenericMesh::GetWireVertexIndexes() {
    // on first call, we init the array
    if (!s_vertIndexesInitialized) {
        unsigned* _vertWireIndexes = s_vertWireIndexes;
        for (unsigned i = 0; i < ccChunk::SIZE * 3; ++i) {
            *_vertWireIndexes++ = i;
            *_vertWireIndexes++ = (((i + 1) % 3) == 0 ? i - 2 : i + 1);
        }
        s_vertIndexesInitialized = true;
    }

    return s_vertWireIndexes;
}

void ccGenericMesh::handleColorRamp(CC_DRAW_CONTEXT& context) {
    if (MACRO_Draw2D(context)) {
        if (MACRO_Foreground(context) && !context.sfColorScaleToDisplay) {
            if (sfShown()) {
                ccGenericPointCloud* vertices = getAssociatedCloud();
                if (!vertices || !vertices->isA(CV_TYPES::POINT_CLOUD)) return;

                ccPointCloud* cloud = static_cast<ccPointCloud*>(vertices);

                // we just need to 'display' the current SF scale if the
                // vertices cloud is hidden (otherwise, it will be taken in
                // charge by the cloud itself)
                if (!cloud->sfColorScaleShown() ||
                    (cloud->sfShown() && cloud->isEnabled() &&
                     cloud->isVisible()))
                    return;

                // we must also check that the parent is not a mesh itself with
                // the same vertices! (in which case it will also take that in
                // charge)
                ccHObject* parent = getParent();
                if (parent && parent->isKindOf(CV_TYPES::MESH) &&
                    (ccHObjectCaster::ToGenericMesh(parent)
                             ->getAssociatedCloud() == vertices))
                    return;

                cloud->addColorRampInfo(context);
                // cloud->drawScale(context);
            }
        }
    }
}

void ccGenericMesh::drawMeOnly(CC_DRAW_CONTEXT& context) {
    ccGenericPointCloud* vertices = getAssociatedCloud();
    if (!vertices) return;

    handleColorRamp(context);

    if (!ecvDisplayTools::GetMainWindow()) return;

    // 3D pass
    if (MACRO_Draw3D(context)) {
        // any triangle?
        unsigned triNum = size();
        if (triNum == 0) return;

        // L.O.D.
        bool lodEnabled =
                (triNum > context.minLODTriangleCount &&
                 context.decimateMeshOnMove && MACRO_LODActivated(context));
        unsigned decimStep =
                (lodEnabled ? static_cast<unsigned>(
                                      ceil(static_cast<double>(triNum * 3) /
                                           context.minLODTriangleCount))
                            : 1);
        unsigned displayedTriNum = triNum / decimStep;

        // display parameters
        glDrawParams glParams;
        getDrawingParameters(glParams);
        // no normals shading without light!
        glParams.showNorms &= bool(MACRO_LightIsEnabled(context));

        // vertices visibility
        const ccGenericPointCloud::VisibilityTableType& verticesVisibility =
                vertices->getTheVisibilityArray();
        bool visFiltering = (verticesVisibility.size() == vertices->size());
        context.visFiltering = visFiltering;

        // wireframe ? (not compatible with LOD)
        bool showWired = isShownAsWire() && !lodEnabled;
        bool isShowPoints = isShownAsPoints() && !lodEnabled;
        if (showWired) {
            context.meshRenderingMode = MESH_RENDERING_MODE::ECV_WIREFRAME_MODE;
        }

        if (isShowPoints) {
            context.meshRenderingMode = MESH_RENDERING_MODE::ECV_POINTS_MODE;
        }
        if (!showWired && !isShowPoints) {
            context.meshRenderingMode = MESH_RENDERING_MODE::ECV_SURFACE_MODE;
        }

        // per-triangle normals?
        bool showTriNormals = (hasTriNormals() && triNormsShown());
        // fix 'showNorms'
        glParams.showNorms = showTriNormals ||
                             (vertices->hasNormals() && m_normalsDisplayed);

        // materials & textures
        bool applyMaterials = (hasMaterials() && materialsShown());
        bool showTextures = (hasTextures() && materialsShown() && !lodEnabled);

        // GL name pushing
        bool pushName = MACRO_DrawEntityNames(context);
        if (pushName) {
            // not fast at all!
            if (MACRO_DrawFastNamesOnly(context)) return;
            // glFunc->glPushName(getUniqueIDForDisplay());
            // minimal display for picking mode!
            glParams.showNorms = false;
            glParams.showColors = false;
            // glParams.showSF --> we keep it only if SF 'NaN' values are
            // hidden
            showTriNormals = false;
            applyMaterials = false;
            showTextures = false;
        }

        // in the case we need to display scalar field colors
        ccScalarField* currentDisplayedScalarField = nullptr;
        bool greyForNanScalarValues = true;
        // unsigned colorRampSteps = 0;
        ccColorScale::Shared colorScale(nullptr);

        if (glParams.showSF) {
            assert(vertices->isA(CV_TYPES::POINT_CLOUD));
            ccPointCloud* cloud = static_cast<ccPointCloud*>(vertices);

            greyForNanScalarValues = (cloud->getCurrentDisplayedScalarField() &&
                                      cloud->getCurrentDisplayedScalarField()
                                              ->areNaNValuesShownInGrey());
            if (greyForNanScalarValues && pushName) {
                // in picking mode, no need to take SF into account if we
                // don't hide any points!
                glParams.showSF = false;
            }
        }

        if (glParams.showColors) {
            if (isColorOverridden()) {
                context.defaultMeshColor = m_tempColor;
            } else {
                assert(vertices->isA(CV_TYPES::POINT_CLOUD));
                context.defaultMeshColor = static_cast<ccPointCloud*>(vertices)
                                                   ->rgbColors()
                                                   ->getValue(0);
            }
        } else {
            context.defaultMeshColor = ecvColor::lightGrey;
        }

        context.drawParam = glParams;

        ecvDisplayTools::Draw(context, this);
    }
}

bool ccGenericMesh::toFile_MeOnly(QFile& out) const {
    if (!ccHObject::toFile_MeOnly(out)) return false;

    //'show wired' state (dataVersion>=20)
    if (out.write(reinterpret_cast<const char*>(&m_showWired), sizeof(bool)) <
        0)
        return WriteError();

    //	//'show points' state (dataVersion>=20)
    //	if (out.write(reinterpret_cast<const char*>(&m_showPoints),
    // sizeof(bool)) < 0) 		return WriteError();

    //'per-triangle normals shown' state (dataVersion>=29))
    if (out.write(reinterpret_cast<const char*>(&m_triNormsShown),
                  sizeof(bool)) < 0)
        return WriteError();

    //'materials shown' state (dataVersion>=29))
    if (out.write(reinterpret_cast<const char*>(&m_materialsShown),
                  sizeof(bool)) < 0)
        return WriteError();

    //'polygon stippling' state (dataVersion>=29))
    if (out.write(reinterpret_cast<const char*>(&m_stippling), sizeof(bool)) <
        0)
        return WriteError();

    return true;
}

bool ccGenericMesh::fromFile_MeOnly(QFile& in,
                                    short dataVersion,
                                    int flags,
                                    LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    //'show wired' state (dataVersion>=20)
    if (in.read(reinterpret_cast<char*>(&m_showWired), sizeof(bool)) < 0)
        return ReadError();

    //	//'show points' state (dataVersion>=20)
    //	if (in.read(reinterpret_cast<char*>(&m_showPoints), sizeof(bool)) < 0)
    //		return ReadError();

    //'per-triangle normals shown' state (dataVersion>=29))
    if (dataVersion >= 29) {
        if (in.read(reinterpret_cast<char*>(&m_triNormsShown), sizeof(bool)) <
            0)
            return ReadError();

        //'materials shown' state (dataVersion>=29))
        if (in.read(reinterpret_cast<char*>(&m_materialsShown), sizeof(bool)) <
            0)
            return ReadError();

        //'polygon stippling' state (dataVersion>=29))
        if (in.read(reinterpret_cast<char*>(&m_stippling), sizeof(bool)) < 0)
            return ReadError();
    }

    return true;
}

ccPointCloud* ccGenericMesh::samplePoints(
        bool densityBased,
        double samplingParameter,
        bool withNormals,
        bool withRGB,
        bool withTexture,
        cloudViewer::GenericProgressCallback* pDlg /*=nullptr*/) {
    if (samplingParameter <= 0) {
        assert(false);
        return nullptr;
    }

    bool withFeatures = (withNormals || withRGB || withTexture);

    QScopedPointer<std::vector<unsigned>> triIndices;
    if (withFeatures) {
        triIndices.reset(new std::vector<unsigned>);
    }

    cloudViewer::PointCloud* sampledCloud = nullptr;
    if (densityBased) {
        sampledCloud = cloudViewer::MeshSamplingTools::samplePointsOnMesh(
                this, samplingParameter, pDlg, triIndices.data());
    } else {
        sampledCloud = cloudViewer::MeshSamplingTools::samplePointsOnMesh(
                this, static_cast<unsigned>(samplingParameter), pDlg,
                triIndices.data());
    }

    // convert to real point cloud
    ccPointCloud* cloud = nullptr;

    if (sampledCloud) {
        if (sampledCloud->size() == 0) {
            CVLog::Warning(
                    "[ccGenericMesh::samplePoints] No point was generated "
                    "(sampling density is too low?)");
        } else {
            cloud = ccPointCloud::From(sampledCloud);
            if (!cloud) {
                CVLog::Warning(
                        "[ccGenericMesh::samplePoints] Not enough memory!");
            }
        }

        delete sampledCloud;
        sampledCloud = nullptr;
    } else {
        CVLog::Warning("[ccGenericMesh::samplePoints] Not enough memory!");
    }

    if (!cloud) {
        return nullptr;
    }

    if (withFeatures && triIndices && triIndices->size() >= cloud->size()) {
        // generate normals
        if (withNormals && hasNormals()) {
            if (cloud->reserveTheNormsTable()) {
                for (unsigned i = 0; i < cloud->size(); ++i) {
                    unsigned triIndex = triIndices->at(i);
                    const CCVector3* P = cloud->getPoint(i);

                    CCVector3 N(0, 0, 1);
                    interpolateNormals(triIndex, *P, N);
                    cloud->addNorm(N);
                }

                cloud->showNormals(true);
            } else {
                CVLog::Warning(
                        "[ccGenericMesh::samplePoints] Failed to interpolate "
                        "normals (not enough memory?)");
            }
        }

        // generate colors
        if (withTexture && hasMaterials()) {
            if (cloud->reserveTheRGBTable()) {
                for (unsigned i = 0; i < cloud->size(); ++i) {
                    unsigned triIndex = triIndices->at(i);
                    const CCVector3* P = cloud->getPoint(i);

                    ecvColor::Rgb C;
                    getColorFromMaterial(triIndex, *P, C, withRGB);
                    cloud->addRGBColor(C);
                }

                cloud->showColors(true);
            } else {
                CVLog::Warning(
                        "[ccGenericMesh::samplePoints] Failed to export "
                        "texture colors (not enough memory?)");
            }
        } else if (withRGB && hasColors()) {
            if (cloud->reserveTheRGBTable()) {
                for (unsigned i = 0; i < cloud->size(); ++i) {
                    unsigned triIndex = triIndices->at(i);
                    const CCVector3* P = cloud->getPoint(i);

                    ecvColor::Rgb C;
                    interpolateColors(triIndex, *P, C);
                    cloud->addRGBColor(C);
                }

                cloud->showColors(true);
            } else {
                CVLog::Warning(
                        "[ccGenericMesh::samplePoints] Failed to interpolate "
                        "colors (not enough memory?)");
            }
        }
    }

    // we rename the resulting cloud
    cloud->setName(getName() + QString(".sampled"));

    // import parameters from both the source vertices and the source mesh
    cloud->copyGlobalShiftAndScale(*this);
    cloud->setGLTransformationHistory(getGLTransformationHistory());

    return cloud;
}

void ccGenericMesh::importParametersFrom(const ccGenericMesh* mesh) {
    if (!mesh) {
        assert(false);
        return;
    }

    // original shift & scale
    copyGlobalShiftAndScale(*mesh);

    // stippling
    enableStippling(mesh->stipplingEnabled());

    // wired style
    showWired(mesh->isShownAsWire());

    // points style
    showPoints(mesh->isShownAsPoints());

    // keep the transformation history!
    setGLTransformationHistory(mesh->getGLTransformationHistory());
    // and meta-data
    setMetaData(mesh->metaData());
}

void ccGenericMesh::computeInterpolationWeights(unsigned triIndex,
                                                const CCVector3& P,
                                                CCVector3d& weights) const {
    cloudViewer::GenericTriangle* tri =
            const_cast<ccGenericMesh*>(this)->_getTriangle(triIndex);
    const CCVector3* A = tri->_getA();
    const CCVector3* B = tri->_getB();
    const CCVector3* C = tri->_getC();

    // barcyentric intepolation weights
    weights.x = ((P - *B).cross(*C - *B)).normd() /*/2*/;
    weights.y = ((P - *C).cross(*A - *C)).normd() /*/2*/;
    weights.z = ((P - *A).cross(*B - *A)).normd() /*/2*/;

    // normalize weights
    double sum = weights.x + weights.y + weights.z;
    weights /= sum;
}

bool ccGenericMesh::trianglePicking(
        unsigned triIndex,
        const CCVector2d& clickPos,
        const ccGLMatrix& trans,
        bool noGLTrans,
        const ccGenericPointCloud& vertices,
        const ccGLCameraParameters& camera,
        CCVector3d& point,
        CCVector3d* barycentricCoords /*=nullptr*/) const {
    assert(triIndex < size());

    CCVector3 A3D;
    CCVector3 B3D;
    CCVector3 C3D;
    getTriangleVertices(triIndex, A3D, B3D, C3D);

    CCVector3d A2D;
    CCVector3d B2D;
    CCVector3d C2D;
    bool inFrustum = true;
    if (noGLTrans) {
        // if none of its points fall into the frustrum the triangle is not
        // visible...
        // DGM: we need to project ALL the points in case at least one is
        // visible
        bool insideA = camera.project(A3D, A2D, &inFrustum);
        bool insideB = camera.project(B3D, B2D, &inFrustum);
        bool insideC = camera.project(C3D, C2D, &inFrustum);
        if (!insideA && !insideB && !insideC) {
            return false;
        }
    } else {
        CCVector3 A3Dp = trans * A3D;
        CCVector3 B3Dp = trans * B3D;
        CCVector3 C3Dp = trans * C3D;
        // if none of its points fall into the frustrum the triangle is not
        // visible...
        // DGM: we need to project ALL the points in case at least one is
        // visible
        bool insideA = camera.project(A3Dp, A2D, &inFrustum);
        bool insideB = camera.project(B3Dp, B2D, &inFrustum);
        bool insideC = camera.project(C3Dp, C2D, &inFrustum);
        if (!insideA && !insideB && !insideC) {
            return false;
        }
    }

    // barycentric coordinates
    GLdouble detT = (B2D.y - C2D.y) * (A2D.x - C2D.x) +
                    (C2D.x - B2D.x) * (A2D.y - C2D.y);
    if (cloudViewer::LessThanEpsilon(std::abs(detT))) {
        return false;
    }
    GLdouble l1 = ((B2D.y - C2D.y) * (clickPos.x - C2D.x) +
                   (C2D.x - B2D.x) * (clickPos.y - C2D.y)) /
                  detT;
    GLdouble l2 = ((C2D.y - A2D.y) * (clickPos.x - C2D.x) +
                   (A2D.x - C2D.x) * (clickPos.y - C2D.y)) /
                  detT;

    // does the point falls inside the triangle?
    if (l1 >= 0 && l1 <= 1.0 && l2 >= 0.0 && l2 <= 1.0) {
        double l1l2 = l1 + l2;
        assert(l1l2 >= -1.0e-12);
        if (l1l2 > 1.0) {
            // we fall outside of the triangle!
            return false;
        }

        GLdouble l3 = 1.0 - l1 - l2;
        assert(l3 >= -1.0e-12);

        // now deduce the 3D position
        point = CCVector3d(l1 * A3D.x + l2 * B3D.x + l3 * C3D.x,
                           l1 * A3D.y + l2 * B3D.y + l3 * C3D.y,
                           l1 * A3D.z + l2 * B3D.z + l3 * C3D.z);

        if (barycentricCoords) {
            *barycentricCoords = CCVector3d(l1, l2, l3);
        }

        return true;
    } else {
        return false;
    }
}

bool ccGenericMesh::trianglePicking(
        const CCVector2d& clickPos,
        const ccGLCameraParameters& camera,
        int& nearestTriIndex,
        double& nearestSquareDist,
        CCVector3d& nearestPoint,
        CCVector3d* barycentricCoords /*=nullptr*/) const {
    ccGLMatrix trans;
    bool noGLTrans = !getAbsoluteGLTransformation(trans);

    // back project the clicked point in 3D
    CCVector3d clickPosd(clickPos.x, clickPos.y, 0);
    CCVector3d X(0, 0, 0);
    if (!camera.unproject(clickPosd, X)) {
        return false;
    }

    nearestTriIndex = -1;
    nearestSquareDist = -1.0;
    nearestPoint = CCVector3d(0, 0, 0);
    if (barycentricCoords) *barycentricCoords = CCVector3d(0, 0, 0);

    ccGenericPointCloud* vertices = getAssociatedCloud();
    if (!vertices) {
        assert(false);
        return false;
    }

#if defined(_OPENMP) && !defined(_DEBUG)
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(size()); ++i) {
        CCVector3d P;
        CCVector3d BC;
        if (!trianglePicking(i, clickPos, trans, noGLTrans, *vertices, camera,
                             P, barycentricCoords ? &BC : nullptr))
            continue;

        double squareDist = (X - P).norm2d();
        if (nearestTriIndex < 0 || squareDist < nearestSquareDist) {
            nearestSquareDist = squareDist;
            nearestTriIndex = static_cast<int>(i);
            nearestPoint = P;
            if (barycentricCoords) *barycentricCoords = BC;
        }
    }

    return (nearestTriIndex >= 0);
}

bool ccGenericMesh::trianglePicking(
        unsigned triIndex,
        const CCVector2d& clickPos,
        const ccGLCameraParameters& camera,
        CCVector3d& point,
        CCVector3d* barycentricCoords /*=nullptr*/) const {
    if (triIndex >= size()) {
        assert(false);
        return false;
    }

    ccGLMatrix trans;
    bool noGLTrans = !getAbsoluteGLTransformation(trans);

    ccGenericPointCloud* vertices = getAssociatedCloud();
    if (!vertices) {
        assert(false);
        return false;
    }

    return trianglePicking(triIndex, clickPos, trans, noGLTrans, *vertices,
                           camera, point, barycentricCoords);
}

bool ccGenericMesh::computePointPosition(
        unsigned triIndex,
        const CCVector2d& uv,
        CCVector3& P,
        bool warningIfOutside /*=true*/) const {
    if (triIndex >= size()) {
        assert(false);
        CVLog::Warning("Index out of range");
        return true;
    }

    CCVector3 A;
    CCVector3 B;
    CCVector3 C;
    getTriangleVertices(triIndex, A, B, C);

    double z = 1.0 - uv.x - uv.y;
    if (warningIfOutside && ((z < -1.0e-6) || (z > 1.0 + 1.0e-6))) {
        CVLog::Warning("Point falls outside of the triangle");
    }

    P = CCVector3(
            static_cast<PointCoordinateType>(uv.x * A.x + uv.y * B.x + z * C.x),
            static_cast<PointCoordinateType>(uv.x * A.y + uv.y * B.y + z * C.y),
            static_cast<PointCoordinateType>(uv.x * A.z + uv.y * B.z +
                                             z * C.z));

    return true;
}

void ccGenericMesh::setGlobalShift(const CCVector3d& shift) {
    if (getAssociatedCloud()) {
        // auto transfer the global shift info to the vertices
        getAssociatedCloud()->setGlobalShift(shift);
    } else {
        // we normally don't want to store this information at
        // the mesh level as it won't be saved.
        assert(false);
        ccShiftedObject::setGlobalShift(shift);
    }
}

void ccGenericMesh::setGlobalScale(double scale) {
    if (getAssociatedCloud()) {
        // auto transfer the global scale info to the vertices
        getAssociatedCloud()->setGlobalScale(scale);
    } else {
        // we normally don't want to store this information at
        // the mesh level as it won't be saved.
        assert(false);
        ccShiftedObject::setGlobalScale(scale);
    }
}

const CCVector3d& ccGenericMesh::getGlobalShift() const {
    return (getAssociatedCloud() ? getAssociatedCloud()->getGlobalShift()
                                 : ccShiftedObject::getGlobalShift());
}

double ccGenericMesh::getGlobalScale() const {
    return (getAssociatedCloud() ? getAssociatedCloud()->getGlobalScale()
                                 : ccShiftedObject::getGlobalScale());
}

bool ccGenericMesh::updateTextures(const std::string& texture_file) {
    std::vector<std::string> texture_files = {texture_file};
    return updateTextures(texture_files);
}

bool ccGenericMesh::updateTextures(
        const std::vector<std::string>& texture_files) {
    if (texture_files.empty()) {
        return false;
    }

    // materials & textures
    bool applyMaterials = (hasMaterials() && materialsShown());
    bool showTextures = (hasTextures() && materialsShown());
    if (applyMaterials || showTextures) {
        auto* materials = const_cast<ccMaterialSet*>(getMaterialSet());
        bool has_materials = true;
        if (!materials) {
            has_materials = false;
            // try to create the materials
            materials = new ccMaterialSet("materials");
            materials->link();
        }

        for (std::size_t ti = 0; ti < texture_files.size(); ++ti) {
            QString textureFileName =
                    QFileInfo(texture_files[ti].c_str()).fileName();
            QString textureFilePath = texture_files[ti].c_str();
            if (ti >= materials->size()) {
                ccMaterial::Shared material(new ccMaterial(textureFileName));
                if (material->loadAndSetTexture(textureFilePath)) {
                    const QImage& texture = material->getTexture();
                    CVLog::Print(QString("[Texture] Successfully loaded "
                                         "texture '%1' (%2x%3 pixels)")
                                         .arg(textureFileName)
                                         .arg(texture.width())
                                         .arg(texture.height()));
                    material->setDiffuse(ecvColor::bright);
                    material->setSpecular(ecvColor::darker);
                    material->setAmbient(ecvColor::light);
                    materials->push_back(material);
                } else {
                    CVLog::Warning(QString("[Texture] Failed to load "
                                           "texture '%1'")
                                           .arg(textureFilePath));
                }
            } else {
                ccMaterial* material =
                        const_cast<ccMaterial*>(materials->at(ti).get());
                assert(material);
                material->releaseTexture();
                material->setName(textureFileName);
                if (material->loadAndSetTexture(textureFilePath)) {
                    const QImage& texture = material->getTexture();
                    CVLog::Print(QString("[Texture] Update "
                                         "texture '%1' (%2x%3 pixels)")
                                         .arg(textureFileName)
                                         .arg(texture.width())
                                         .arg(texture.height()));
                } else {
                    CVLog::Warning(QString("[Texture] Failed to load "
                                           "texture '%1'")
                                           .arg(textureFilePath));
                }
            }
        }

        if (!has_materials) {
            int childIndex = getChildIndex(materials);
            if (childIndex < 0) addChild(materials);
        }

        CC_DRAW_CONTEXT context;
        context.viewID = getViewId();
        ecvDisplayTools::UpdateMeshTextures(context, this);
        return true;
    } else {
        return false;
    }
}

bool ccGenericMesh::IsCloudVerticesOfMesh(ccGenericPointCloud* cloud,
                                          ccGenericMesh** mesh /*=nullptr*/) {
    if (!cloud) {
        assert(false);
        return false;
    }

    // check whether the input point cloud acts as the vertices of a mesh
    {
        ccHObject* parent = cloud->getParent();
        if (parent && parent->isKindOf(CV_TYPES::MESH) &&
            static_cast<ccGenericMesh*>(parent)->getAssociatedCloud() ==
                    cloud) {
            if (mesh) {
                *mesh = static_cast<ccGenericMesh*>(parent);
            }
            return true;
        }
    }

    // now check the children
    for (unsigned i = 0; i < cloud->getChildrenNumber(); ++i) {
        ccHObject* child = cloud->getChild(i);
        if (child && child->isKindOf(CV_TYPES::MESH) &&
            static_cast<ccGenericMesh*>(child)->getAssociatedCloud() == cloud) {
            if (mesh) {
                *mesh = static_cast<ccGenericMesh*>(child);
            }
            return true;
        }
    }

    return false;
}
