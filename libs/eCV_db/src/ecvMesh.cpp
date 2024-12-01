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

#include "ecvMesh.h"

// Local
#include "ecvBBox.h"
#include "ecvChunk.h"
#include "ecvColorScalesManager.h"
#include "ecvDisplayTools.h"
#include "ecvGenericPointCloud.h"
#include "ecvHObjectCaster.h"
#include "ecvMaterialSet.h"
#include "ecvNormalVectors.h"
#include "ecvOrientedBBox.h"
#include "ecvPointCloud.h"
#include "ecvPolyline.h"
#include "ecvProgressDialog.h"
#include "ecvScalarField.h"
#include "ecvSubMesh.h"

// cloudViewer
#include <Delaunay2dMesh.h>
#include <Logging.h>
#include <ManualSegmentationTools.h>
#include <Neighbourhood.h>
#include <PointProjectionTools.h>
#include <ReferenceCloud.h>

// System
#include <assert.h>
#include <string.h>

#include <cmath>  //for std::modf

static CCVector3 s_blankNorm(0, 0, 0);

ccMesh::ccMesh(ccGenericPointCloud* vertices)
    : ccGenericMesh("Mesh"),
      m_associatedCloud(nullptr),
      m_triNormals(nullptr),
      m_texCoords(nullptr),
      m_materials(nullptr),
      m_triVertIndexes(nullptr),
      m_globalIterator(0),
      m_triMtlIndexes(nullptr),
      m_texCoordIndexes(nullptr),
      m_triNormalIndexes(nullptr) {
    setAssociatedCloud(vertices);

    m_triVertIndexes = new triangleIndexesContainer();
    m_triVertIndexes->link();
}

ccMesh::ccMesh(const ccMesh& mesh) : ccMesh(new ccPointCloud("vertices")) {
    if (m_associatedCloud && getChildrenNumber() == 0) {
        m_associatedCloud->setEnabled(false);
        // DGM: no need to lock it as it is only used by one mesh!
        m_associatedCloud->setLocked(false);
        this->addChild(m_associatedCloud);
    }

    *this = mesh;
}

ccMesh::ccMesh(const std::vector<Eigen::Vector3d>& vertices,
               const std::vector<Eigen::Vector3i>& triangles)
    : ccMesh(new ccPointCloud("vertices")) {
    assert(m_associatedCloud);
    m_associatedCloud->setEnabled(false);
    m_associatedCloud->setLocked(false);
    // DGM: no need to lock it as it is only used by one mesh!
    this->addChild(m_associatedCloud);
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    if (cloud->resize(static_cast<unsigned>(vertices.size()))) {
        this->setEigenVertices(vertices);
    }

    this->setTriangles(triangles);
}

ccMesh::ccMesh(cloudViewer::GenericIndexedMesh* giMesh,
               ccGenericPointCloud* giVertices)
    : ccGenericMesh("Mesh"),
      m_associatedCloud(nullptr),
      m_triNormals(nullptr),
      m_texCoords(nullptr),
      m_materials(nullptr),
      m_triVertIndexes(nullptr),
      m_globalIterator(0),
      m_triMtlIndexes(nullptr),
      m_texCoordIndexes(nullptr),
      m_triNormalIndexes(nullptr) {
    setAssociatedCloud(giVertices);

    m_triVertIndexes = new triangleIndexesContainer();
    m_triVertIndexes->link();

    unsigned triNum = giMesh->size();
    if (!reserve(triNum)) return;

    giMesh->placeIteratorAtBeginning();
    for (unsigned i = 0; i < triNum; ++i) {
        const cloudViewer::VerticesIndexes* tsi =
                giMesh->getNextTriangleVertIndexes();
        addTriangle(tsi->i1, tsi->i2, tsi->i3);
    }

    // if (!giVertices->hasNormals())
    //	computeNormals();
    showNormals(giVertices->hasNormals());

    if (giVertices->hasColors()) showColors(giVertices->colorsShown());

    if (giVertices->hasDisplayedScalarField()) showSF(giVertices->sfShown());
}

ccMesh::~ccMesh() {
    clearTriNormals();
    setMaterialSet(nullptr);
    setTexCoordinatesTable(nullptr);

    if (m_triVertIndexes) m_triVertIndexes->release();
    if (m_texCoordIndexes) m_texCoordIndexes->release();
    if (m_triMtlIndexes) m_triMtlIndexes->release();
    if (m_triNormalIndexes) m_triNormalIndexes->release();
}

ccMesh& ccMesh::operator=(const ccMesh& mesh) {
    if (!getAssociatedCloud()) {
        ccPointCloud* baseVertices = new ccPointCloud("vertices");
        baseVertices->setEnabled(false);
        // DGM: no need to lock it as it is only used by one mesh!
        baseVertices->setLocked(false);
        setAssociatedCloud(baseVertices);
        this->setName("Merged mesh");
        this->addChild(baseVertices);

    } else {
        ccHObjectCaster::ToPointCloud(getAssociatedCloud())->clear();
    }

    this->resize(0);

    this->adjacency_list_ = mesh.adjacency_list_;

    this->merge(&mesh, false);
    return (*this);
}

ccMesh& ccMesh::operator+=(const ccMesh& mesh) {
    if (!this->merge(&mesh, false)) {
        CVLog::Error("Fusion failed! (not enough memory?)");
    }

    this->adjacency_list_ = mesh.adjacency_list_;

    return (*this);
}

ccMesh ccMesh::operator+(const ccMesh& mesh) const {
    return (ccMesh(*this) += mesh);
}

void ccMesh::setAssociatedCloud(ccGenericPointCloud* cloud) {
    m_associatedCloud = cloud;

    if (m_associatedCloud)
        m_associatedCloud->addDependency(
                this, DP_NOTIFY_OTHER_ON_DELETE | DP_NOTIFY_OTHER_ON_UPDATE);

    m_bBox.setValidity(false);
}

bool ccMesh::createInternalCloud() {
    if (getAssociatedCloud()) {
        if (getChildrenNumber() == 0) {
            addChild(getAssociatedCloud());
        }
        cloudViewer::utility::LogWarning(
                "Already has associated vertices cloud!");
        return false;
    }

    ccPointCloud* cloud = new ccPointCloud("vertices");
    if (!cloud) {
        cloudViewer::utility::LogWarning(
                "Creating associated vertices cloud failed!");
        return false;
    }
    cloud->setEnabled(false);
    cloud->setLocked(false);
    // DGM: no need to lock it as it is only used by one mesh!
    setAssociatedCloud(cloud);

    if (getChildrenNumber() == 0) {
        this->addChild(cloud);
    }
    return true;
}

void ccMesh::onUpdateOf(ccHObject* obj) {
    if (obj == m_associatedCloud) {
        m_bBox.setValidity(false);
        notifyGeometryUpdate();  // for sub-meshes
    }

    ccGenericMesh::onUpdateOf(obj);
}

void ccMesh::onDeletionOf(const ccHObject* obj) {
    if (obj == m_associatedCloud) setAssociatedCloud(nullptr);

    ccGenericMesh::onDeletionOf(obj);
}

bool ccMesh::hasColors() const {
    return (m_associatedCloud ? m_associatedCloud->hasColors() : false);
}

bool ccMesh::hasNormals() const {
    return ((m_associatedCloud ? m_associatedCloud->hasNormals() : false) ||
            hasTriNormals());
}

bool ccMesh::hasDisplayedScalarField() const {
    return (m_associatedCloud ? m_associatedCloud->hasDisplayedScalarField()
                              : false);
}

bool ccMesh::hasScalarFields() const {
    return (m_associatedCloud ? m_associatedCloud->hasScalarFields() : false);
}

bool ccMesh::computeNormals(bool perVertex) {
    return perVertex ? computePerVertexNormals() : computePerTriangleNormals();
}

bool ccMesh::computePerVertexNormals() {
    if (!m_associatedCloud ||
        !m_associatedCloud->isA(CV_TYPES::POINT_CLOUD))  // TODO
    {
        CVLog::Warning(
                "[ccMesh::computePerVertexNormals] Vertex set is not a "
                "standard cloud?!");
        return false;
    }

    unsigned triCount = size();
    if (triCount == 0) {
        CVLog::Warning("[ccMesh::computePerVertexNormals] Empty mesh!");
        return false;
    }
    unsigned vertCount = m_associatedCloud->size();
    if (vertCount < 3) {
        CVLog::Warning(
                "[ccMesh::computePerVertexNormals] Not enough vertices! (<3)");
        return false;
    }

    ccPointCloud* cloud = static_cast<ccPointCloud*>(m_associatedCloud);

    // we instantiate a temporary structure to store each vertex normal
    // (uncompressed)
    std::vector<CCVector3> theNorms;
    try {
        theNorms.resize(vertCount, s_blankNorm);
    } catch (const std::bad_alloc&) {
        CVLog::Warning("[ccMesh::computePerVertexNormals] Not enough memory!");
        return false;
    }

    // allocate compressed normals array on vertices cloud
    bool normalsWereAllocated = cloud->hasNormals();
    if (/*!normalsWereAllocated && */ !cloud
                ->resizeTheNormsTable())  // we call it whatever the case (just
                                          // to be sure)
    {
        // warning message should have been already issued!
        return false;
    }

    // for each triangle
    placeIteratorAtBeginning();
    {
        for (unsigned i = 0; i < triCount; ++i) {
            cloudViewer::VerticesIndexes* tsi = getNextTriangleVertIndexes();

            assert(tsi->i1 < vertCount && tsi->i2 < vertCount &&
                   tsi->i3 < vertCount);
            const CCVector3* A = cloud->getPoint(tsi->i1);
            const CCVector3* B = cloud->getPoint(tsi->i2);
            const CCVector3* C = cloud->getPoint(tsi->i3);

            // compute face normal (right hand rule)
            CCVector3 N = (*B - *A).cross(*C - *A);
            // N.normalize(); //DGM: no normalization = weighting by surface!

            // we add this normal to all triangle vertices
            theNorms[tsi->i1] += N;
            theNorms[tsi->i2] += N;
            theNorms[tsi->i3] += N;
        }
    }

    // for each vertex
    {
        for (unsigned i = 0; i < vertCount; i++) {
            CCVector3& N = theNorms[i];
            // normalize the 'mean' normal
            N.normalize();
            cloud->setPointNormal(i, N);
        }
    }

    // apply it also to sub-meshes!
    showNormals_extended(true);

    if (!normalsWereAllocated) cloud->showNormals(true);

    return true;
}

bool ccMesh::computePerTriangleNormals() {
    unsigned triCount = size();
    if (triCount == 0) {
        CVLog::Warning("[ccMesh::computePerTriangleNormals] Empty mesh!");
        return false;
    }

    // if some normal indexes already exists, we remove them (easier)
    if (m_triNormalIndexes) removePerTriangleNormalIndexes();
    setTriNormsTable(nullptr);

    NormsIndexesTableType* normIndexes = new NormsIndexesTableType();
    if (!normIndexes->reserveSafe(triCount)) {
        normIndexes->release();
        CVLog::Warning(
                "[ccMesh::computePerTriangleNormals] Not enough memory!");
        return false;
    }

    // for each triangle
    {
        for (unsigned i = 0; i < triCount; ++i) {
            const cloudViewer::VerticesIndexes& tri =
                    m_triVertIndexes->getValue(i);
            const CCVector3* A = m_associatedCloud->getPoint(tri.i1);
            const CCVector3* B = m_associatedCloud->getPoint(tri.i2);
            const CCVector3* C = m_associatedCloud->getPoint(tri.i3);

            // compute face normal (right hand rule)
            CCVector3 N = (*B - *A).cross(*C - *A);

            CompressedNormType nIndex = ccNormalVectors::GetNormIndex(N.u);
            normIndexes->emplace_back(nIndex);
        }
    }

    // set the per-triangle normal indexes
    {
        if (!reservePerTriangleNormalIndexes()) {
            normIndexes->release();
            CVLog::Warning(
                    "[ccMesh::computePerTriangleNormals] Not enough memory!");
            return false;
        }

        setTriNormsTable(normIndexes);

        for (int i = 0; i < static_cast<int>(triCount); ++i)
            addTriangleNormalIndexes(i, i, i);
    }

    // apply it also to sub-meshes!
    showNormals_extended(true);

    return true;
}

bool ccMesh::normalsShown() const {
    return (ccHObject::normalsShown() || triNormsShown());
}

bool ccMesh::processScalarField(MESH_SCALAR_FIELD_PROCESS process) {
    if (!m_associatedCloud || !m_associatedCloud->isScalarFieldEnabled())
        return false;

    unsigned nPts = m_associatedCloud->size();

    // instantiate memory for per-vertex mean SF
    ScalarType* meanSF = new ScalarType[nPts];
    if (!meanSF) {
        // Not enough memory!
        return false;
    }

    // per-vertex counters
    unsigned* count = new unsigned[nPts];
    if (!count) {
        // Not enough memory!
        delete[] meanSF;
        return false;
    }

    // init arrays
    {
        for (unsigned i = 0; i < nPts; ++i) {
            meanSF[i] = m_associatedCloud->getPointScalarValue(i);
            count[i] = 1;
        }
    }

    // for each triangle
    unsigned nTri = size();
    {
        placeIteratorAtBeginning();
        for (unsigned i = 0; i < nTri; ++i) {
            const cloudViewer::VerticesIndexes* tsi =
                    getNextTriangleVertIndexes();  // DGM:
                                                   // getNextTriangleVertIndexes
                                                   // is faster for mesh groups!

            // compute the sum of all connected vertices SF values
            meanSF[tsi->i1] += m_associatedCloud->getPointScalarValue(tsi->i2);
            meanSF[tsi->i2] += m_associatedCloud->getPointScalarValue(tsi->i3);
            meanSF[tsi->i3] += m_associatedCloud->getPointScalarValue(tsi->i1);

            // TODO DGM: we could weight this by the vertices distance?
            ++count[tsi->i1];
            ++count[tsi->i2];
            ++count[tsi->i3];
        }
    }

    // normalize
    {
        for (unsigned i = 0; i < nPts; ++i) meanSF[i] /= count[i];
    }

    switch (process) {
        case SMOOTH_MESH_SF: {
            // Smooth = mean value
            for (unsigned i = 0; i < nPts; ++i)
                m_associatedCloud->setPointScalarValue(i, meanSF[i]);
        } break;
        case ENHANCE_MESH_SF: {
            // Enhance = old value + (old value - mean value)
            for (unsigned i = 0; i < nPts; ++i) {
                ScalarType v = 2 * m_associatedCloud->getPointScalarValue(i) -
                               meanSF[i];
                m_associatedCloud->setPointScalarValue(i, v > 0 ? v : 0);
            }
        } break;
    }

    delete[] meanSF;
    delete[] count;

    return true;
}

void ccMesh::setTriNormsTable(NormsIndexesTableType* triNormsTable,
                              bool autoReleaseOldTable /*=true*/) {
    if (m_triNormals == triNormsTable) return;

    if (m_triNormals && autoReleaseOldTable) {
        int childIndex = getChildIndex(m_triNormals);
        m_triNormals->release();
        m_triNormals = nullptr;
        if (childIndex >= 0) removeChild(childIndex);
    }

    m_triNormals = triNormsTable;
    if (m_triNormals) {
        m_triNormals->link();
        int childIndex = getChildIndex(m_triNormals);
        if (childIndex < 0) addChild(m_triNormals);
    } else {
        removePerTriangleNormalIndexes();  // auto-remove per-triangle indexes
                                           // (we don't need them anymore)
    }
}

void ccMesh::setMaterialSet(ccMaterialSet* materialSet,
                            bool autoReleaseOldMaterialSet /*=true*/) {
    if (m_materials == materialSet) return;

    if (m_materials && autoReleaseOldMaterialSet) {
        int childIndex = getChildIndex(m_materials);
        m_materials->release();
        m_materials = nullptr;
        if (childIndex >= 0) removeChild(childIndex);
    }

    m_materials = materialSet;
    if (m_materials) {
        m_materials->link();
        int childIndex = getChildIndex(m_materials);
        if (childIndex < 0) addChild(m_materials);
    } else {
        removePerTriangleMtlIndexes();  // auto-remove per-triangle indexes (we
                                        // don't need them anymore)
    }
}

void ccMesh::applyGLTransformation(const ccGLMatrix& trans) {
    // transparent call
    ccGenericMesh::applyGLTransformation(trans);

    // we take care of per-triangle normals
    //(vertices and per-vertex normals should be taken care of by the recursive
    // call)
    transformTriNormals(trans);
}

void ccMesh::transformTriNormals(const ccGLMatrix& trans) {
    // we must take care of the triangle normals!
    if (m_triNormals &&
        (!getParent() || !getParent()->isKindOf(CV_TYPES::MESH))) {
        size_t numTriNormals = m_triNormals->size();

#if 0  // no use to use memory for this!
	  bool recoded = false;

		//if there are more triangle normals than the size of the compressed
		//normals array, we recompress the array instead of recompressing each normal
		if (numTriNormals > ccNormalVectors::GetNumberOfVectors())
		{
			NormsIndexesTableType* newNorms = new NormsIndexesTableType;
			if (newNorms->reserve(ccNormalVectors::GetNumberOfVectors()))
			{
				//decode
				{
					for (unsigned i=0; i<ccNormalVectors::GetNumberOfVectors(); i++)
					{
						CCVector3 new_n(ccNormalVectors::GetNormal(i));
						trans.applyRotation(new_n);
						CompressedNormType newNormIndex = ccNormalVectors::GetNormIndex(new_n.u);
						newNorms->emplace_back(newNormIndex);
					}
				}

				//recode
				m_triNormals->placeIteratorAtBeginning();
				{
					for (unsigned i=0; i<numTriNormals; i++)
					{
						m_triNormals->setValue(i,newNorms->getValue(m_triNormals->getCurrentValue()));
						m_triNormals->forwardIterator();
					}
				}
				recoded = true;
			}
			newNorms->clear();
			newNorms->release();
			newNorms = 0;
		}

		//if there are less triangle normals than the compressed normals array size
		//(or if there is not enough memory to instantiate the temporary array),
		//we recompress each normal ...
		if (!recoded)
#endif
        {
            for (CompressedNormType& _theNormIndex : *m_triNormals) {
                CCVector3 new_n(ccNormalVectors::GetNormal(_theNormIndex));
                trans.applyRotation(new_n);
                _theNormIndex = ccNormalVectors::GetNormIndex(new_n.u);
            }
        }
    }
}

static bool TagDuplicatedVertices(
        const cloudViewer::DgmOctree::octreeCell& cell,
        void** additionalParameters,
        cloudViewer::NormalizedProgress* nProgress /*=0*/) {
    std::vector<int>* equivalentIndexes =
            static_cast<std::vector<int>*>(additionalParameters[0]);

    // we look for points very close to the others (only if not yet tagged!)

    // structure for nearest neighbors search
    cloudViewer::DgmOctree::NearestNeighboursSphericalSearchStruct nNSS;
    nNSS.level = cell.level;
    static const PointCoordinateType c_defaultSearchRadius =
            static_cast<PointCoordinateType>(sqrtf(ZERO_TOLERANCE_F));
    nNSS.prepare(c_defaultSearchRadius,
                 cell.parentOctree->getCellSize(nNSS.level));
    cell.parentOctree->getCellPos(cell.truncatedCode, cell.level, nNSS.cellPos,
                                  true);
    cell.parentOctree->computeCellCenter(nNSS.cellPos, cell.level,
                                         nNSS.cellCenter);

    unsigned n = cell.points->size();  // number of points in the current cell

    // we already know some of the neighbours: the points in the current cell!
    try {
        nNSS.pointsInNeighbourhood.resize(n);
    } catch (... /*const std::bad_alloc&*/)  // out of memory
    {
        return false;
    }

    // init structure with cell points
    {
        cloudViewer::DgmOctree::NeighboursSet::iterator it =
                nNSS.pointsInNeighbourhood.begin();
        for (unsigned i = 0; i < n; ++i, ++it) {
            it->point = cell.points->getPointPersistentPtr(i);
            it->pointIndex = cell.points->getPointGlobalIndex(i);
        }
        nNSS.alreadyVisitedNeighbourhoodSize = 1;
    }

    // for each point in the cell
    for (unsigned i = 0; i < n; ++i) {
        int thisIndex = static_cast<int>(cell.points->getPointGlobalIndex(i));
        if (equivalentIndexes->at(thisIndex) < 0)  // has no equivalent yet
        {
            cell.points->getPoint(i, nNSS.queryPoint);

            // look for neighbors in a (very small) sphere
            // warning: there may be more points at the end of
            // nNSS.pointsInNeighbourhood than the actual nearest neighbors (k)!
            unsigned k =
                    cell.parentOctree->findNeighborsInASphereStartingFromCell(
                            nNSS, c_defaultSearchRadius, false);

            // if there are some very close points
            if (k > 1) {
                for (unsigned j = 0; j < k; ++j) {
                    // all the other points are equivalent to the query point
                    const unsigned& otherIndex =
                            nNSS.pointsInNeighbourhood[j].pointIndex;
                    if (static_cast<int>(otherIndex) != thisIndex)
                        equivalentIndexes->at(otherIndex) = thisIndex;
                }
            }

            // and the query point is always root
            equivalentIndexes->at(thisIndex) = thisIndex;
        }

        if (nProgress && !nProgress->oneStep()) {
            return false;
        }
    }

    return true;
}

bool ccMesh::mergeDuplicatedVertices(unsigned char octreeLevel /*=10*/,
                                     QWidget* parentWidget /*=nullptr*/) {
    if (!m_associatedCloud) {
        assert(false);
        return false;
    }

    unsigned vertCount = m_associatedCloud->size();
    unsigned faceCount = size();
    if (vertCount == 0 || faceCount == 0) {
        CVLog::Warning(
                "[ccMesh::mergeDuplicatedVertices] No triangle or no vertex");
        return false;
    }

    try {
        std::vector<int> equivalentIndexes;
        const int razValue = -1;
        equivalentIndexes.resize(vertCount, razValue);

        // tag the duplicated vertices
        {
            QScopedPointer<ecvProgressDialog> pDlg(nullptr);
            if (parentWidget) {
                pDlg.reset(new ecvProgressDialog(true, parentWidget));
            }

            // try to build the octree
            ccOctree::Shared octree =
                    ccOctree::Shared(new ccOctree(m_associatedCloud));
            if (!octree->build(pDlg.data())) {
                CVLog::Warning("[MergeDuplicatedVertices] Not enough memory");
                return false;
            }

            void* additionalParameters[] = {
                    static_cast<void*>(&equivalentIndexes)};
            unsigned result = octree->executeFunctionForAllCellsAtLevel(
                    10, TagDuplicatedVertices, additionalParameters, false,
                    pDlg.data(), "Tag duplicated vertices");

            if (result == 0) {
                CVLog::Warning(
                        "[MergeDuplicatedVertices] Duplicated vertices removal "
                        "algorithm failed?!");
                return false;
            }
        }

        unsigned remainingCount = 0;
        for (unsigned i = 0; i < vertCount; ++i) {
            int eqIndex = equivalentIndexes[i];
            assert(eqIndex >= 0);
            if (eqIndex == static_cast<int>(i))  // root point
            {
                // we replace the root index by its 'new' index (+ vertCount, to
                // differentiate it later)
                int newIndex = static_cast<int>(vertCount + remainingCount);
                equivalentIndexes[i] = newIndex;
                ++remainingCount;
            }
        }

        cloudViewer::ReferenceCloud newVerticesRef(m_associatedCloud);
        if (!newVerticesRef.reserve(remainingCount)) {
            CVLog::Warning("[MergeDuplicatedVertices] Not enough memory");
            return false;
        }

        // copy root points in a new cloud
        {
            for (unsigned i = 0; i < vertCount; ++i) {
                int eqIndex = equivalentIndexes[i];
                if (eqIndex >= static_cast<int>(vertCount))  // root point
                    newVerticesRef.addPointIndex(i);
                else
                    equivalentIndexes[i] =
                            equivalentIndexes[eqIndex];  // and update the other
                                                         // indexes
            }
        }

        ccPointCloud* newVertices = nullptr;
        if (m_associatedCloud->isKindOf(CV_TYPES::POINT_CLOUD)) {
            newVertices = static_cast<ccPointCloud*>(m_associatedCloud)
                                  ->partialClone(&newVerticesRef);
        } else {
            newVertices =
                    ccPointCloud::From(&newVerticesRef, m_associatedCloud);
        }
        if (!newVertices) {
            CVLog::Warning("[MergeDuplicatedVertices] Not enough memory");
            return false;
        }

        // update face indexes
        {
            unsigned newFaceCount = 0;
            for (unsigned i = 0; i < faceCount; ++i) {
                cloudViewer::VerticesIndexes* tri = getTriangleVertIndexes(i);
                tri->i1 = static_cast<unsigned>(equivalentIndexes[tri->i1]) -
                          vertCount;
                tri->i2 = static_cast<unsigned>(equivalentIndexes[tri->i2]) -
                          vertCount;
                tri->i3 = static_cast<unsigned>(equivalentIndexes[tri->i3]) -
                          vertCount;

                // very small triangles (or flat ones) may be implicitly removed
                // by vertex fusion!
                if (tri->i1 != tri->i2 && tri->i1 != tri->i3 &&
                    tri->i2 != tri->i3) {
                    if (newFaceCount != i) swapTriangles(i, newFaceCount);
                    ++newFaceCount;
                }
            }

            if (newFaceCount == 0) {
                CVLog::Warning(
                        "[MergeDuplicatedVertices] After vertex fusion, all "
                        "triangles would collapse! We'll keep the non-fused "
                        "version...");
                delete newVertices;
                newVertices = nullptr;
            } else {
                resize(newFaceCount);
            }
        }

        // update the mesh vertices
        int childPos = getChildIndex(m_associatedCloud);
        if (childPos >= 0) {
            removeChild(childPos);
        } else {
            delete m_associatedCloud;
            m_associatedCloud = nullptr;
        }
        setAssociatedCloud(newVertices);
        if (childPos >= 0) {
            addChild(m_associatedCloud);
        }
        vertCount = m_associatedCloud->size();
        CVLog::Print(
                "[MergeDuplicatedVertices] Remaining vertices after "
                "auto-removal of duplicate ones: %i",
                vertCount);
        CVLog::Print(
                "[MergeDuplicatedVertices] Remaining faces after auto-removal "
                "of duplicate ones: %i",
                size());
        return false;
    } catch (const std::bad_alloc&) {
        CVLog::Warning(
                "[MergeDuplicatedVertices] Not enough memory: could not remove "
                "duplicated vertices!");
    }

    return false;
}

Eigen::Vector3d ccMesh::getMinBound() const {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(getAssociatedCloud());
    if (!cloud) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return ComputeMinBound(cloud->getEigenPoints());
}

Eigen::Vector3d ccMesh::getMaxBound() const {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(getAssociatedCloud());
    if (!cloud) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return ComputeMaxBound(cloud->getEigenPoints());
}

Eigen::Vector3d ccMesh::getGeometryCenter() const {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(getAssociatedCloud());
    if (!cloud) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return ComputeCenter(cloud->getEigenPoints());
}

ccBBox ccMesh::getAxisAlignedBoundingBox() const {
    return ccBBox::CreateFromPoints(getVertices());
}

ecvOrientedBBox ccMesh::getOrientedBoundingBox() const {
    return ecvOrientedBBox::CreateFromPoints(getVertices());
}

ccMesh& ccMesh::transform(const Eigen::Matrix4d& transformation) {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    if (!cloud) {
        return *this;
    }
    ccGLMatrix mat = ccGLMatrix::FromEigenMatrix(transformation);
    cloud->applyRigidTransformation(mat);
    transformTriNormals(mat);
    return *this;
}

ccMesh& ccMesh::translate(const Eigen::Vector3d& translation, bool relative) {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    if (!cloud) {
        return *this;
    }
    cloud->translate(translation, relative);
    return *this;
}

ccMesh& ccMesh::scale(const double s, const Eigen::Vector3d& center) {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    if (!cloud) {
        return *this;
    }

    cloud->scale(s, center);
    return *this;
}

ccMesh& ccMesh::rotate(const Eigen::Matrix3d& R,
                       const Eigen::Vector3d& center) {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
    if (!cloud) {
        return *this;
    }

    ccGLMatrix trans;
    trans.setRotation(R.data());
    transformTriNormals(trans);

    trans.shiftRotationCenter(center);
    cloud->applyRigidTransformation(trans);
    return *this;
}

std::shared_ptr<ccMesh> ccMesh::selectByIndex(
        const std::vector<size_t>& indices, bool cleanup) const {
    if (hasTriangleUvs()) {
        cloudViewer::utility::LogWarning(
                "[SelectByIndices] This mesh contains triangle uvs that are "
                "not handled in this function");
    }

    ccPointCloud* baseVertices = new ccPointCloud("vertices");
    assert(baseVertices);

    auto output = cloudViewer::make_shared<ccMesh>(baseVertices);
    output->reserve(indices.size());
    bool has_triangle_normals = hasTriNormals();
    bool has_vertex_normals = hasNormals();
    bool has_vertex_colors = hasColors();

    std::vector<int> new_vert_ind(getVerticeSize(), -1);
    for (const auto& sel_vidx : indices) {
        if (sel_vidx < 0 || sel_vidx >= getVerticeSize()) {
            cloudViewer::utility::LogWarning(
                    "[SelectByIndex] indices contains index {} out of range. "
                    "It is ignored.",
                    sel_vidx);
            continue;
        }
        if (new_vert_ind[sel_vidx] >= 0) {
            continue;
        }
        new_vert_ind[sel_vidx] = int(output->getVerticeSize());

        baseVertices->addPoint(*(getAssociatedCloud()->getPoint(
                static_cast<unsigned>(sel_vidx))));
        if (has_vertex_normals) {
            if (!baseVertices->hasNormals()) {
                baseVertices->reserveTheNormsTable();
            }
            baseVertices->addNorm(getAssociatedCloud()->getPointNormal(
                    static_cast<unsigned>(sel_vidx)));
        }
        if (has_vertex_colors) {
            if (!baseVertices->hasColors()) {
                baseVertices->reserveTheRGBTable();
            }
            baseVertices->addRGBColor(getAssociatedCloud()->getPointColor(
                    static_cast<unsigned>(sel_vidx)));
        }
    }

    for (size_t tidx = 0; tidx < size(); ++tidx) {
        Eigen::Vector3i triangle;
        getTriangleVertIndexes(tidx, triangle);
        int nvidx0 = new_vert_ind[triangle(0)];
        int nvidx1 = new_vert_ind[triangle(1)];
        int nvidx2 = new_vert_ind[triangle(2)];
        if (nvidx0 >= 0 && nvidx1 >= 0 && nvidx2 >= 0) {
            output->addTriangle(Eigen::Vector3i(nvidx0, nvidx1, nvidx2));
            if (has_triangle_normals) {
                output->addTriangleNorm(getTriangleNorm(tidx));
            }
        }
    }

    // do some cleaning
    {
        if (cleanup) {
            output->removeDuplicatedVertices();
            output->removeDuplicatedTriangles();
            output->removeUnreferencedVertices();
            output->removeDegenerateTriangles();
        }

        baseVertices->shrinkToFit();
        output->shrinkToFit();
        NormsIndexesTableType* normals = output->getTriNormsTable();
        if (normals) {
            normals->shrink_to_fit();
        }
    }

    baseVertices->setEnabled(false);
    // DGM: no need to lock it as it is only used by one mesh!
    baseVertices->setLocked(false);
    output->addChild(baseVertices);

    cloudViewer::utility::LogDebug(
            "Triangle mesh sampled from {:d} vertices and {:d} triangles to "
            "{:d} vertices and {:d} triangles.",
            (int)getVerticeSize(), (int)size(), (int)output->getVerticeSize(),
            (int)output->size());

    return output;
}

std::shared_ptr<ccMesh> ccMesh::crop(const ccBBox& bbox) const {
    if (!bbox.isValid()) {
        cloudViewer::utility::LogError(
                "[ccMesh::crop] ccBBox either has zeros "
                "size, or has wrong bounds.");
        return cloudViewer::make_shared<ccMesh>(nullptr);
    }
    return selectByIndex(bbox.getPointIndicesWithinBoundingBox(getVertices()));
}

std::shared_ptr<ccMesh> ccMesh::crop(const ecvOrientedBBox& bbox) const {
    if (bbox.isEmpty()) {
        cloudViewer::utility::LogError(
                "[ccMesh::crop] ecvOrientedBBox either has zeros "
                "size, or has wrong bounds.");
        return cloudViewer::make_shared<ccMesh>(nullptr);
    }
    return selectByIndex(bbox.getPointIndicesWithinBoundingBox(
            ccHObjectCaster::ToPointCloud(m_associatedCloud)->getPoints()));
}

bool ccMesh::laplacianSmooth(unsigned nbIteration,
                             PointCoordinateType factor,
                             ecvProgressDialog* progressCb /*=0*/) {
    if (!m_associatedCloud) return false;

    // vertices
    unsigned vertCount = m_associatedCloud->size();
    // triangles
    unsigned faceCount = size();
    if (!vertCount || !faceCount) return false;

    std::vector<CCVector3> verticesDisplacement;
    try {
        verticesDisplacement.resize(vertCount);
    } catch (const std::bad_alloc&) {
        // not enough memory
        return false;
    }

    // compute the number of edges to which belong each vertex
    std::vector<unsigned> edgesCount;
    try {
        edgesCount.resize(vertCount, 0);
    } catch (const std::bad_alloc&) {
        // not enough memory
        return false;
    }

    placeIteratorAtBeginning();
    for (unsigned j = 0; j < faceCount; j++) {
        const cloudViewer::VerticesIndexes* tri = getNextTriangleVertIndexes();
        edgesCount[tri->i1] += 2;
        edgesCount[tri->i2] += 2;
        edgesCount[tri->i3] += 2;
    }

    // progress dialog
    cloudViewer::NormalizedProgress nProgress(progressCb, nbIteration);
    if (progressCb) {
        progressCb->setMethodTitle(QObject::tr("Laplacian smooth"));
        progressCb->setInfo(
                QObject::tr("Iterations: %1\nVertices: %2\nFaces: %3")
                        .arg(nbIteration)
                        .arg(vertCount)
                        .arg(faceCount));
        progressCb->start();
    }

    // repeat Laplacian smoothing iterations
    for (unsigned iter = 0; iter < nbIteration; iter++) {
        std::fill(verticesDisplacement.begin(), verticesDisplacement.end(),
                  CCVector3(0, 0, 0));

        // for each triangle
        placeIteratorAtBeginning();
        for (unsigned j = 0; j < faceCount; j++) {
            const cloudViewer::VerticesIndexes* tri =
                    getNextTriangleVertIndexes();

            const CCVector3* A = m_associatedCloud->getPoint(tri->i1);
            const CCVector3* B = m_associatedCloud->getPoint(tri->i2);
            const CCVector3* C = m_associatedCloud->getPoint(tri->i3);

            CCVector3 dAB = (*B - *A);
            CCVector3 dAC = (*C - *A);
            CCVector3 dBC = (*C - *B);

            verticesDisplacement[tri->i1] += dAB + dAC;
            verticesDisplacement[tri->i2] += dBC - dAB;
            verticesDisplacement[tri->i3] -= dAC + dBC;
        }

        if (!nProgress.oneStep()) {
            // cancelled by user
            break;
        }

        // apply displacement
        for (unsigned i = 0; i < vertCount; i++) {
            if (edgesCount[i]) {
                // this is a "persistent" pointer and we know what type of cloud
                // is behind ;)
                CCVector3* P = const_cast<CCVector3*>(
                        m_associatedCloud->getPointPersistentPtr(i));
                (*P) += verticesDisplacement[i] * (factor / edgesCount[i]);
            }
        }
    }

    m_associatedCloud->notifyGeometryUpdate();

    if (hasNormals()) computeNormals(!hasTriNormals());

    return true;
}

ccMesh* ccMesh::cloneMesh(ccGenericPointCloud* vertices /*=0*/,
                          ccMaterialSet* clonedMaterials /*=0*/,
                          NormsIndexesTableType* clonedNormsTable /*=0*/,
                          TextureCoordsContainer* cloneTexCoords /*=0*/) {
    assert(m_associatedCloud);

    // vertices
    unsigned vertNum = m_associatedCloud->size();
    // triangles
    unsigned triNum = size();

    // temporary structure to check that vertices are really used (in case of
    // vertices set sharing)
    std::vector<unsigned> usedVerts;

    ccGenericPointCloud* newVertices = vertices;

    // no input vertices set
    if (!newVertices) {
        // let's check the real vertex count
        try {
            usedVerts.resize(vertNum, 0);
        } catch (const std::bad_alloc&) {
            CVLog::Error("[ccMesh::clone] Not enough memory!");
            return nullptr;
        }

        // flag used vertices
        {
            placeIteratorAtBeginning();
            for (unsigned i = 0; i < triNum; ++i) {
                const cloudViewer::VerticesIndexes* tsi =
                        getNextTriangleVertIndexes();
                usedVerts[tsi->i1] = 1;
                usedVerts[tsi->i2] = 1;
                usedVerts[tsi->i3] = 1;
            }
        }

        // we check that all points in 'associatedCloud' are used by this mesh
        unsigned realVertCount = 0;
        {
            for (unsigned i = 0; i < vertNum; ++i)
                usedVerts[i] = (usedVerts[i] == 1 ? realVertCount++ : vertNum);
        }

        // the associated cloud is already the exact vertices set --> nothing to
        // change
        if (realVertCount == vertNum) {
            newVertices = m_associatedCloud->clone(nullptr, true);
        } else {
            // we create a temporary entity with used vertices only
            cloudViewer::ReferenceCloud rc(m_associatedCloud);
            if (rc.reserve(realVertCount)) {
                for (unsigned i = 0; i < vertNum; ++i) {
                    if (usedVerts[i] != vertNum)
                        rc.addPointIndex(i);  // can't fail, see above
                }

                // and the associated vertices set
                assert(m_associatedCloud->isA(CV_TYPES::POINT_CLOUD));
                newVertices = static_cast<ccPointCloud*>(m_associatedCloud)
                                      ->partialClone(&rc);
                if (newVertices && newVertices->size() < rc.size()) {
                    // not enough memory!
                    delete newVertices;
                    newVertices = nullptr;
                }
            }
        }
    }

    // failed to create a new vertices set!
    if (!newVertices) {
        CVLog::Error("[ccMesh::clone] Not enough memory!");
        return nullptr;
    }

    // mesh clone
    ccMesh* cloneMesh = new ccMesh(newVertices);
    if (!cloneMesh->reserve(triNum)) {
        if (!vertices) delete newVertices;
        delete cloneMesh;
        CVLog::Error("[ccMesh::clone] Not enough memory!");
        return nullptr;
    }

    // let's create the new triangles
    if (!usedVerts.empty())  // in case we have an equivalence table
    {
        placeIteratorAtBeginning();
        for (unsigned i = 0; i < triNum; ++i) {
            const cloudViewer::VerticesIndexes* tsi =
                    getNextTriangleVertIndexes();
            cloneMesh->addTriangle(usedVerts[tsi->i1], usedVerts[tsi->i2],
                                   usedVerts[tsi->i3]);
        }
        usedVerts.resize(0);
    } else {
        placeIteratorAtBeginning();
        for (unsigned i = 0; i < triNum; ++i) {
            const cloudViewer::VerticesIndexes* tsi =
                    getNextTriangleVertIndexes();
            cloneMesh->addTriangle(tsi->i1, tsi->i2, tsi->i3);
        }
    }

    // triangle normals
    if (m_triNormals && m_triNormalIndexes) {
        // 1st: try to allocate per-triangle normals indexes
        if (cloneMesh->reservePerTriangleNormalIndexes()) {
            // 2nd: clone the main array if not already done
            if (!clonedNormsTable) {
                clonedNormsTable =
                        m_triNormals
                                ->clone();  // TODO: keep only what's necessary!
                if (clonedNormsTable)
                    cloneMesh->addChild(clonedNormsTable);
                else {
                    CVLog::Warning(
                            "[ccMesh::clone] Not enough memory: failed to "
                            "clone per-triangle normals!");
                    cloneMesh->removePerTriangleNormalIndexes();  // don't need
                                                                  // this
                                                                  // anymore!
                }
            }

            // if we have both the main array and per-triangle normals indexes,
            // we can finish the job
            if (cloneMesh) {
                cloneMesh->setTriNormsTable(clonedNormsTable);
                assert(cloneMesh->m_triNormalIndexes);
                m_triNormalIndexes->copy(
                        *cloneMesh->m_triNormalIndexes);  // should be ok as
                                                          // array is already
                                                          // reserved!
            }
        } else {
            CVLog::Warning(
                    "[ccMesh::clone] Not enough memory: failed to clone "
                    "per-triangle normal indexes!");
        }
    }

    // materials
    if (m_materials && m_triMtlIndexes) {
        // 1st: try to allocate per-triangle materials indexes
        if (cloneMesh->reservePerTriangleMtlIndexes()) {
            // 2nd: clone the main array if not already done
            if (!clonedMaterials) {
                clonedMaterials =
                        getMaterialSet()
                                ->clone();  // TODO: keep only what's necessary!
                if (clonedMaterials) {
                    cloneMesh->addChild(clonedMaterials);
                } else {
                    CVLog::Warning(
                            "[ccMesh::clone] Not enough memory: failed to "
                            "clone materials set!");
                    cloneMesh->removePerTriangleMtlIndexes();  // don't need
                                                               // this anymore!
                }
            }

            // if we have both the main array and per-triangle materials
            // indexes, we can finish the job
            if (clonedMaterials) {
                cloneMesh->setMaterialSet(clonedMaterials);
                assert(cloneMesh->m_triMtlIndexes);
                m_triMtlIndexes->copy(
                        *cloneMesh->m_triMtlIndexes);  // should be ok as array
                                                       // is already reserved!
            }
        } else {
            CVLog::Warning(
                    "[ccMesh::clone] Not enough memory: failed to clone "
                    "per-triangle materials!");
        }
    }

    // texture coordinates
    if (m_texCoords && m_texCoordIndexes) {
        // 1st: try to allocate per-triangle texture info
        if (cloneMesh->reservePerTriangleTexCoordIndexes()) {
            // 2nd: clone the main array if not already done
            if (!cloneTexCoords) {
                cloneTexCoords =
                        m_texCoords
                                ->clone();  // TODO: keep only what's necessary!
                if (!cloneTexCoords) {
                    CVLog::Warning(
                            "[ccMesh::clone] Not enough memory: failed to "
                            "clone texture coordinates!");
                    cloneMesh->removePerTriangleTexCoordIndexes();  // don't
                                                                    // need this
                                                                    // anymore!
                }
            }

            // if we have both the main array and per-triangle texture info, we
            // can finish the job
            if (cloneTexCoords) {
                cloneMesh->setTexCoordinatesTable(cloneTexCoords);
                assert(cloneMesh->m_texCoordIndexes);
                m_texCoordIndexes->copy(
                        *cloneMesh
                                 ->m_texCoordIndexes);  // should be ok as array
                                                        // is already reserved!
            }
        } else {
            CVLog::Warning(
                    "[ccMesh::clone] Not enough memory: failed to clone "
                    "per-triangle texture info!");
        }
    }

    if (!vertices) {
        if (hasNormals() && !cloneMesh->hasNormals())
            cloneMesh->computeNormals(!hasTriNormals());
        newVertices->setEnabled(false);
        // we link the mesh structure with the new vertex set
        cloneMesh->addChild(newVertices);
        // cloneMesh->setDisplay_recursive(getDisplay());
    }

    cloneMesh->showNormals(normalsShown());
    cloneMesh->showColors(colorsShown());
    cloneMesh->showSF(sfShown());
    cloneMesh->showMaterials(materialsShown());
    cloneMesh->setName(getName() + QString(".clone"));
    cloneMesh->setVisible(isVisible());
    cloneMesh->setEnabled(isEnabled());
    cloneMesh->importParametersFrom(this);

    return cloneMesh;
}

ccMesh* ccMesh::TriangulateTwoPolylines(ccPolyline* p1,
                                        ccPolyline* p2,
                                        CCVector3* projectionDir /*=0*/) {
    if (!p1 || p1->size() == 0 || !p2 || p2->size() == 0) {
        assert(false);
        return nullptr;
    }

    ccPointCloud* vertices = new ccPointCloud("vertices");
    if (!vertices->reserve(p1->size() + p2->size())) {
        CVLog::Warning("[ccMesh::TriangulateTwoPolylines] Not enough memory");
        delete vertices;
        return nullptr;
    }

    // merge the two sets of vertices
    {
        for (unsigned i = 0; i < p1->size(); ++i)
            vertices->addPoint(*p1->getPoint(i));
        for (unsigned j = 0; j < p2->size(); ++j)
            vertices->addPoint(*p2->getPoint(j));
    }
    assert(vertices->size() != 0);

    cloudViewer::Neighbourhood N(vertices);

    // get plane coordinate system
    CCVector3 O = *N.getGravityCenter();
    CCVector3 X(1, 0, 0), Y(0, 1, 0);
    if (projectionDir) {
        // use the input projection dir.
        X = projectionDir->orthogonal();
        Y = projectionDir->cross(X);
    } else {
        // use the best fit plane (normal)
        if (!N.getLSPlane()) {
            CVLog::Warning(
                    "[ccMesh::TriangulateTwoPolylines] Failed to fit a plane "
                    "through both polylines");
            delete vertices;
            return nullptr;
        }

        X = *N.getLSPlaneX();
        Y = *N.getLSPlaneY();
    }

    std::vector<CCVector2> points2D;
    std::vector<int> segments2D;
    try {
        points2D.reserve(p1->size() + p2->size());
        segments2D.reserve(p1->segmentCount() + p2->segmentCount());
    } catch (const std::bad_alloc&) {
        // not enough memory
        CVLog::Warning("[ccMesh::TriangulateTwoPolylines] Not enough memory");
        delete vertices;
        return nullptr;
    }

    // project the polylines on the best fitting plane
    {
        ccPolyline* polylines[2] = {p1, p2};
        for (size_t i = 0; i < 2; ++i) {
            ccPolyline* poly = polylines[i];
            unsigned vertCount = poly->size();
            int vertIndex0 = static_cast<int>(points2D.size());
            bool closed = poly->isClosed();
            for (unsigned v = 0; v < vertCount; ++v) {
                const CCVector3* P = poly->getPoint(v);
                int vertIndex = static_cast<int>(points2D.size());

                CCVector3 OP = *P - O;
                CCVector2 P2D(OP.dot(X), OP.dot(Y));
                points2D.emplace_back(P2D);

                if (v + 1 < vertCount) {
                    segments2D.emplace_back(vertIndex);
                    segments2D.emplace_back(vertIndex + 1);
                } else if (closed) {
                    segments2D.emplace_back(vertIndex);
                    segments2D.emplace_back(vertIndex0);
                }
            }
        }
        assert(points2D.size() == p1->size() + p2->size());
        assert(segments2D.size() ==
               (p1->segmentCount() + p2->segmentCount()) * 2);
    }

    cloudViewer::Delaunay2dMesh* delaunayMesh = new cloudViewer::Delaunay2dMesh;
    std::string errorStr;
    if (!delaunayMesh->buildMesh(points2D, segments2D, errorStr)) {
        CVLog::Warning(QString("Third party library error: %1")
                               .arg(QString::fromStdString(errorStr)));
        delete delaunayMesh;
        delete vertices;
        return nullptr;
    }

    delaunayMesh->linkMeshWith(vertices, false);

    // remove the points oustide of the 'concave' hull
    {
        // first compute the Convex hull
        std::vector<cloudViewer::PointProjectionTools::IndexedCCVector2>
                indexedPoints2D;
        try {
            indexedPoints2D.resize(points2D.size());
            for (size_t i = 0; i < points2D.size(); ++i) {
                indexedPoints2D[i] = points2D[i];
                indexedPoints2D[i].index = static_cast<unsigned>(i);
            }

            std::list<cloudViewer::PointProjectionTools::IndexedCCVector2*>
                    hullPoints;
            if (cloudViewer::PointProjectionTools::extractConvexHull2D(
                        indexedPoints2D, hullPoints)) {
                std::list<cloudViewer::PointProjectionTools::
                                  IndexedCCVector2*>::iterator A =
                        hullPoints.begin();
                for (; A != hullPoints.end(); ++A) {
                    // current hull segment
                    std::list<cloudViewer::PointProjectionTools::
                                      IndexedCCVector2*>::iterator B = A;
                    ++B;
                    if (B == hullPoints.end()) {
                        B = hullPoints.begin();
                    }

                    unsigned Aindex = (*A)->index;
                    unsigned Bindex = (*B)->index;
                    int Apoly = (Aindex < p1->size() ? 0 : 1);
                    int Bpoly = (Bindex < p1->size() ? 0 : 1);
                    // both vertices belong to the same polyline
                    if (Apoly == Bpoly) {
                        // if it creates an outer loop
                        if (abs(static_cast<int>(Bindex) -
                                static_cast<int>(Aindex)) > 1) {
                            // create the corresponding contour
                            unsigned iStart = std::min(Aindex, Bindex);
                            unsigned iStop = std::max(Aindex, Bindex);
                            std::vector<CCVector2> contour;
                            contour.reserve(iStop - iStart + 1);
                            for (unsigned j = iStart; j <= iStop; ++j) {
                                contour.emplace_back(points2D[j]);
                            }
                            delaunayMesh->removeOuterTriangles(
                                    points2D, contour,
                                    /*remove inside = */ false);
                        }
                    }
                }
            } else {
                CVLog::Warning(
                        "[ccMesh::TriangulateTwoPolylines] Failed to compute "
                        "the convex hull (can't clean the mesh borders)");
            }
        } catch (const std::bad_alloc&) {
            CVLog::Warning(
                    "[ccMesh::TriangulateTwoPolylines] Not enough memory to "
                    "clean the mesh borders");
        }
    }

    ccMesh* mesh = new ccMesh(delaunayMesh, vertices);
    if (mesh->size() != delaunayMesh->size()) {
        // not enough memory (error will be issued later)
        delete mesh;
        mesh = nullptr;
    }

    // don't need this anymore
    delete delaunayMesh;
    delaunayMesh = nullptr;

    if (mesh) {
        mesh->addChild(vertices);
        mesh->setVisible(true);
        vertices->setEnabled(false);

        // global shift & scale (we copy it from the first polyline by default)
        vertices->setGlobalShift(p1->getGlobalShift());
        vertices->setGlobalScale(p1->getGlobalScale());
        // same thing for the display
        // mesh->setDisplay(p1->getDisplay());
    } else {
        CVLog::Warning("[ccMesh::TriangulateTwoPolylines] Not enough memory");
        delete vertices;
        vertices = nullptr;
    }

    return mesh;
}

ccMesh* ccMesh::Triangulate(ccGenericPointCloud* cloud,
                            cloudViewer::TRIANGULATION_TYPES type,
                            bool updateNormals /*=false*/,
                            PointCoordinateType maxEdgeLength /*=0*/,
                            unsigned char dim /*=2*/) {
    if (!cloud || dim > 2) {
        CVLog::Warning("[ccMesh::Triangulate] Invalid input parameters!");
        return nullptr;
    }
    if (cloud->size() < 3) {
        CVLog::Warning("[ccMesh::Triangulate] Cloud has not enough points!");
        return nullptr;
    }

    // compute raw mesh
    std::string errorStr;
    cloudViewer::GenericIndexedMesh* dummyMesh =
            cloudViewer::PointProjectionTools::computeTriangulation(
                    cloud, type, maxEdgeLength, dim, errorStr);
    if (!dummyMesh) {
        CVLog::Warning(QString("[ccMesh::Triangulate] Failed to construct "
                               "Delaunay mesh (Triangle lib error: %1)")
                               .arg(QString::fromStdString(errorStr)));
        return nullptr;
    }

    // convert raw mesh to ccMesh
    ccMesh* mesh = new ccMesh(dummyMesh, cloud);

    // don't need this anymore
    delete dummyMesh;
    dummyMesh = nullptr;

    if (!mesh) {
        CVLog::Warning(
                "[ccMesh::Triangulate] An error occurred while computing mesh! "
                "(not enough memory?)");
        return nullptr;
    }

    mesh->setName(cloud->getName() + QString(".mesh"));
    // mesh->setDisplay(cloud->getDisplay());
    bool cloudHadNormals = cloud->hasNormals();
    // compute per-vertex normals if necessary
    if (!cloudHadNormals || updateNormals) {
        mesh->computeNormals(true);
    }
    mesh->showNormals(cloudHadNormals || !cloud->hasColors());
    if (mesh->getAssociatedCloud() && mesh->getAssociatedCloud() != cloud) {
        mesh->getAssociatedCloud()->setGlobalShift(cloud->getGlobalShift());
        mesh->getAssociatedCloud()->setGlobalScale(cloud->getGlobalScale());
    }

    return mesh;
}

bool ccMesh::merge(const ccMesh* mesh, bool createSubMesh) {
    if (!mesh) {
        assert(false);
        CVLog::Warning("[ccMesh::merge] Internal error: invalid input!");
        return false;
    }
    if (!mesh->getAssociatedCloud() ||
        !mesh->getAssociatedCloud()->isA(CV_TYPES::POINT_CLOUD) ||
        !m_associatedCloud || !m_associatedCloud->isA(CV_TYPES::POINT_CLOUD)) {
        assert(false);
        CVLog::Warning(
                "[ccMesh::merge] Requires meshes with standard vertices!");
        return false;
    }
    ccPointCloud* vertices =
            static_cast<ccPointCloud*>(mesh->getAssociatedCloud());

    // vertices count (before merge)
    const unsigned vertNumBefore = m_associatedCloud->size();
    // triangles count (before merge)
    const unsigned triNumBefore = size();

    bool success = false;

    for (int iteration = 0; iteration < 1;
         ++iteration)  // fake loop for easy breaking/cleaning
    {
        // merge vertices
        unsigned vertIndexShift = 0;
        if (mesh->getAssociatedCloud() != m_associatedCloud) {
            unsigned vertAdded = mesh->getAssociatedCloud()->size();
            static_cast<ccPointCloud*>(m_associatedCloud)
                    ->append(vertices, m_associatedCloud->size(), true);

            // not enough memory?
            if (m_associatedCloud->size() < vertNumBefore + vertAdded) {
                CVLog::Warning("[ccMesh::merge] Not enough memory!");
                break;
            }
            vertIndexShift = vertNumBefore;
            if (vertNumBefore == 0) {
                // use the first merged cloud display properties
                m_associatedCloud->setVisible(vertices->isVisible());
                m_associatedCloud->setEnabled(vertices->isEnabled());
            }
        }
        showNormals(this->normalsShown() || mesh->normalsShown());
        showColors(this->colorsShown() || mesh->colorsShown());
        showSF(this->sfShown() || mesh->sfShown());

        // now for the triangles
        const unsigned triAdded = mesh->size();
        bool otherMeshHasMaterials =
                (mesh->m_materials && mesh->m_triMtlIndexes);
        bool otherMeshHasTexCoords =
                (mesh->m_texCoords && mesh->m_texCoordIndexes);
        bool otherMeshHasTriangleNormals =
                (mesh->m_triNormals && mesh->m_triNormalIndexes);
        {
            if (!reserve(triNumBefore + triAdded)) {
                CVLog::Warning("[ccMesh::merge] Not enough memory!");
                break;
            }

            // we'll need those arrays later
            if ((/*otherMeshHasMaterials && */ !m_triMtlIndexes &&
                 !reservePerTriangleMtlIndexes()) ||
                (otherMeshHasTexCoords && !m_texCoordIndexes &&
                 !reservePerTriangleTexCoordIndexes()) ||
                (otherMeshHasTriangleNormals && !m_triNormalIndexes &&
                 !reservePerTriangleNormalIndexes())) {
                CVLog::Warning("[ccMesh::merge] Not enough memory!");
                break;
            }

            for (unsigned i = 0; i < triAdded; ++i) {
                const cloudViewer::VerticesIndexes* tsi =
                        mesh->getTriangleVertIndexes(i);
                addTriangle(vertIndexShift + tsi->i1, vertIndexShift + tsi->i2,
                            vertIndexShift + tsi->i3);
            }
        }

        // triangle normals
        bool hasTriangleNormals = m_triNormals && m_triNormalIndexes;
        if (hasTriangleNormals || otherMeshHasTriangleNormals) {
            // 1st: does the other mesh has triangle normals
            if (otherMeshHasTriangleNormals) {
                size_t triIndexShift = 0;
                if (m_triNormals != mesh->m_triNormals) {
                    // reserve mem for triangle normals
                    if (!m_triNormals) {
                        NormsIndexesTableType* normsTable =
                                new NormsIndexesTableType();
                        setTriNormsTable(normsTable);
                    }
                    assert(m_triNormals);
                    size_t triNormalsCountBefore = m_triNormals->size();
                    if (!m_triNormals->reserveSafe(
                                triNormalsCountBefore +
                                mesh->m_triNormals->size())) {
                        CVLog::Warning("[ccMesh::merge] Not enough memory!");
                        break;
                    }
                    // copy the values
                    {
                        for (unsigned i = 0; i < mesh->m_triNormals->size();
                             ++i)
                            m_triNormals->emplace_back(
                                    mesh->m_triNormals->getValue(i));
                    }
                    triIndexShift = triNormalsCountBefore;
                }

                // the indexes should have already been resized by the call to
                // 'reserve'!
                assert(m_triNormalIndexes->capacity() >=
                       triNumBefore + triAdded);
                // copy the values
                {
                    for (unsigned i = 0; i < mesh->m_triNormalIndexes->size();
                         ++i) {
                        const Tuple3i& indexes =
                                mesh->m_triNormalIndexes->at(i);
                        Tuple3i newIndexes(
                                indexes.u[0] < 0
                                        ? -1
                                        : indexes.u[0] + static_cast<int>(
                                                                 triIndexShift),
                                indexes.u[1] < 0
                                        ? -1
                                        : indexes.u[1] + static_cast<int>(
                                                                 triIndexShift),
                                indexes.u[2] < 0
                                        ? -1
                                        : indexes.u[2] +
                                                  static_cast<int>(
                                                          triIndexShift));
                        m_triNormalIndexes->emplace_back(newIndexes);
                    }
                }
            } else {
                // the indexes should have already been resized by the call to
                // 'reserve'!
                assert(m_triNormalIndexes->capacity() >=
                       triNumBefore + triAdded);
                // fill the indexes table with default values
                {
                    Tuple3i defaultElement(-1, -1, -1);
                    for (unsigned i = 0; i < mesh->size(); ++i)
                        m_triNormalIndexes->emplace_back(defaultElement);
                }
            }
            showTriNorms(this->triNormsShown() || mesh->triNormsShown());
        }

        // materials
        bool hasMaterials = m_materials && m_triMtlIndexes;
        if (hasMaterials || otherMeshHasMaterials) {
            // 1st: does the other mesh has materials?
            if (otherMeshHasMaterials) {
                std::vector<int> materialIndexMap;
                if (m_materials != mesh->m_materials) {
                    // reserve mem for materials
                    if (!m_materials) {
                        ccMaterialSet* set = new ccMaterialSet("materials");
                        setMaterialSet(set);
                    }
                    assert(m_materials);

                    size_t otherMatSetSize = mesh->m_materials->size();
                    try {
                        materialIndexMap.resize(otherMatSetSize, -1);
                    } catch (const std::bad_alloc&) {
                        CVLog::Warning("[ccMesh::merge] Not enough memory!");
                        break;
                    }
                    // update map table
                    for (size_t m = 0; m != otherMatSetSize; ++m) {
                        materialIndexMap[m] = m_materials->addMaterial(
                                mesh->m_materials->at(m));
                    }
                }

                // the indexes should have already been resized by the call to
                // 'reserve'!
                assert(m_triMtlIndexes->capacity() >= triNumBefore + triAdded);
                // copy the values
                {
                    for (unsigned i = 0; i < mesh->m_triMtlIndexes->size();
                         ++i) {
                        int index = mesh->m_triMtlIndexes->getValue(i);
                        assert(index <
                               static_cast<int>(materialIndexMap.size()));
                        int newIndex =
                                (index < 0 ? -1 : materialIndexMap[index]);
                        m_triMtlIndexes->emplace_back(newIndex);
                    }
                }
            } else {
                // the indexes should have already been resized by the call to
                // 'reserve'!
                assert(m_triMtlIndexes->capacity() >= triNumBefore + triAdded);
                // fill the indexes table with default values
                {
                    for (unsigned i = 0; i < mesh->size(); ++i)
                        m_triMtlIndexes->emplace_back(-1);
                }
            }
        }
        showMaterials(this->materialsShown() || mesh->materialsShown());

        // texture coordinates
        bool hasTexCoords = m_texCoords && m_texCoordIndexes;
        if (hasTexCoords || otherMeshHasTexCoords) {
            // 1st: does the other mesh has texture coordinates?
            if (otherMeshHasTexCoords) {
                size_t texCoordIndexShift = 0;
                if (m_texCoords != mesh->m_texCoords) {
                    // reserve mem for triangle normals
                    if (!m_texCoords) {
                        TextureCoordsContainer* texCoordsTable =
                                new TextureCoordsContainer;
                        setTexCoordinatesTable(texCoordsTable);
                    }
                    assert(m_texCoords);
                    size_t texCoordCountBefore = m_texCoords->size();
                    if (!m_texCoords->reserveSafe(texCoordCountBefore +
                                                  mesh->m_texCoords->size())) {
                        CVLog::Warning("[ccMesh::merge] Not enough memory!");
                        break;
                    }
                    // copy the values
                    {
                        static const TexCoords2D TxDef(-1.0f, -1.0f);
                        for (unsigned i = 0; i < mesh->m_texCoords->size();
                             ++i) {
                            const TexCoords2D& T = mesh->m_texCoords->at(i);
                            m_texCoords->emplace_back(T);
                        }
                    }
                    texCoordIndexShift = texCoordCountBefore;
                }

                // the indexes should have already been resized by the call to
                // 'reserve'!
                assert(m_texCoordIndexes->capacity() >=
                       triNumBefore + triAdded);
                // copy the values
                {
                    for (unsigned i = 0; i < mesh->m_texCoordIndexes->size();
                         ++i) {
                        const Tuple3i& indexes =
                                mesh->m_texCoordIndexes->getValue(i);
                        Tuple3i newIndexes(
                                indexes.u[0] < 0
                                        ? -1
                                        : indexes.u[0] +
                                                  static_cast<int>(
                                                          texCoordIndexShift),
                                indexes.u[1] < 0
                                        ? -1
                                        : indexes.u[1] +
                                                  static_cast<int>(
                                                          texCoordIndexShift),
                                indexes.u[2] < 0
                                        ? -1
                                        : indexes.u[2] +
                                                  static_cast<int>(
                                                          texCoordIndexShift));
                        m_texCoordIndexes->emplace_back(newIndexes);
                    }
                }
            } else {
                // the indexes should have already been resized by the call to
                // 'reserve'!
                assert(m_texCoordIndexes->capacity() >=
                       triNumBefore + triAdded);
                // fill the indexes table with default values
                {
                    Tuple3i defaultElement(-1, -1, -1);
                    for (unsigned i = 0; i < mesh->m_texCoordIndexes->size();
                         ++i)
                        m_texCoordIndexes->emplace_back(defaultElement);
                }
            }
        }

        // the end!
        showWired(this->isShownAsWire() || mesh->isShownAsWire());
        showPoints(this->isShownAsPoints() || mesh->isShownAsPoints());
        enableStippling(this->stipplingEnabled() || mesh->stipplingEnabled());
        success = true;
    }

    if (createSubMesh) {
        // triangles count (after merge)
        const unsigned triNumAfter = size();

        ccSubMesh* subMesh = new ccSubMesh(this);
        if (subMesh->reserve(triNumAfter - triNumBefore)) {
            subMesh->addTriangleIndex(triNumBefore, triNumAfter);
            subMesh->setName(mesh->getName());
            subMesh->showMaterials(materialsShown());
            subMesh->showNormals(normalsShown());
            subMesh->showTriNorms(triNormsShown());
            subMesh->showColors(colorsShown());
            subMesh->showWired(isShownAsWire());
            subMesh->showPoints(isShownAsPoints());
            subMesh->enableStippling(stipplingEnabled());
            subMesh->setEnabled(false);
            addChild(subMesh);
        } else {
            CVLog::Warning(QString("[Merge] Not enough memory to create the "
                                   "sub-mesh corresponding to mesh '%1'!")
                                   .arg(mesh->getName()));
            delete subMesh;
            subMesh = nullptr;
        }
    }

    // textures and materials
    {
        textures_ = mesh->textures_;
        triangle_uvs_ = mesh->triangle_uvs_;
        adjacency_list_ = mesh->adjacency_list_;
        triangle_material_ids_ = mesh->triangle_material_ids_;
    }

    if (!success) {
        // revert to original state
        static_cast<ccPointCloud*>(m_associatedCloud)->resize(vertNumBefore);
        resize(triNumBefore);
    }

    return success;
}

void ccMesh::clear() {
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(getAssociatedCloud());
    assert(cloud);
    if (cloud && cloud->hasPoints()) {
        cloud->clear();
    }

    resize(0);
    clearTriNormals();

    adjacency_list_.clear();
    triangle_uvs_.clear();
    materials_.clear();
    triangle_material_ids_.clear();
    textures_.clear();
}

unsigned ccMesh::size() const {
    return static_cast<unsigned>(m_triVertIndexes->size());
}

unsigned ccMesh::capacity() const {
    return static_cast<unsigned>(m_triVertIndexes->capacity());
}

void ccMesh::forEach(genericTriangleAction action) {
    if (!m_associatedCloud) return;

    for (unsigned i = 0; i < m_triVertIndexes->size(); ++i) {
        const cloudViewer::VerticesIndexes& tri = m_triVertIndexes->at(i);
        m_currentTriangle.A = m_associatedCloud->getPoint(tri.i1);
        m_currentTriangle.B = m_associatedCloud->getPoint(tri.i2);
        m_currentTriangle.C = m_associatedCloud->getPoint(tri.i3);
        action(m_currentTriangle);
    }
}

void ccMesh::placeIteratorAtBeginning() { m_globalIterator = 0; }

cloudViewer::GenericTriangle* ccMesh::_getNextTriangle() {
    if (m_globalIterator < m_triVertIndexes->size()) {
        return _getTriangle(m_globalIterator++);
    }

    return nullptr;
}

cloudViewer::GenericTriangle* ccMesh::_getTriangle(
        unsigned triangleIndex)  // temporary
{
    assert(triangleIndex < m_triVertIndexes->size());

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triangleIndex);
    m_currentTriangle.A = m_associatedCloud->getPoint(tri.i1);
    m_currentTriangle.B = m_associatedCloud->getPoint(tri.i2);
    m_currentTriangle.C = m_associatedCloud->getPoint(tri.i3);

    return &m_currentTriangle;
}

void ccMesh::getTriangleVertices(unsigned triangleIndex,
                                 CCVector3& A,
                                 CCVector3& B,
                                 CCVector3& C) const {
    assert(triangleIndex < m_triVertIndexes->size());

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triangleIndex);
    m_associatedCloud->getPoint(tri.i1, A);
    m_associatedCloud->getPoint(tri.i2, B);
    m_associatedCloud->getPoint(tri.i3, C);
}

void ccMesh::getTriangleVertices(unsigned triangleIndex,
                                 double A[3],
                                 double B[3],
                                 double C[3]) const {
    assert(triangleIndex < m_triVertIndexes->size());

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triangleIndex);
    m_associatedCloud->getPoint(tri.i1, A);
    m_associatedCloud->getPoint(tri.i2, B);
    m_associatedCloud->getPoint(tri.i3, C);
}

Eigen::Vector3d ccMesh::getVertice(size_t index) const {
    if (!getAssociatedCloud()) {
        return Eigen::Vector3d();
    }

    return ccHObjectCaster::ToPointCloud(getAssociatedCloud())
            ->getEigenPoint(index);
}

void ccMesh::setVertice(size_t index, const Eigen::Vector3d& vertice) {
    if (!getAssociatedCloud()) {
        return;
    }
    ccHObjectCaster::ToPointCloud(getAssociatedCloud())
            ->setEigenPoint(index, vertice);
}

void ccMesh::addVertice(const Eigen::Vector3d& vertice) {
    if (!getAssociatedCloud()) {
        return;
    }
    ccHObjectCaster::ToPointCloud(getAssociatedCloud())->addEigenPoint(vertice);
}

void ccMesh::setVertexNormal(size_t index, const Eigen::Vector3d& normal) {
    if (m_associatedCloud) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        cloud->setPointNormal(index, normal);
    }
}

void ccMesh::addVertexNormal(const Eigen::Vector3d& normal) {
    if (m_associatedCloud) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        cloud->addEigenNorm(normal);
    }
}

Eigen::Vector3d ccMesh::getVertexNormal(size_t index) const {
    if (!m_associatedCloud) {
        return Eigen::Vector3d();
    }
    return ccHObjectCaster::ToPointCloud(m_associatedCloud)
            ->getEigenNormal(index);
}

void ccMesh::setVertexNormals(const std::vector<Eigen::Vector3d>& normals) {
    if (m_associatedCloud && normals.size() == m_associatedCloud->size()) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        cloud->setEigenNormals(normals);
    }
}

std::vector<Eigen::Vector3d> ccMesh::getVertexNormals() const {
    if (!m_associatedCloud) {
        return std::vector<Eigen::Vector3d>();
    }
    return ccHObjectCaster::ToPointCloud(m_associatedCloud)->getEigenNormals();
}

void ccMesh::addVertexNormals(const std::vector<Eigen::Vector3d>& normals) {
    if (m_associatedCloud) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        if (cloud->reserveTheNormsTable()) {
            cloud->addEigenNorms(normals);
        }
    }
}

void ccMesh::setVertexColor(size_t index, const Eigen::Vector3d& color) {
    if (m_associatedCloud) {
        ccHObjectCaster::ToPointCloud(m_associatedCloud)
                ->setPointColor(index, color);
    }
}

void ccMesh::addVertexColor(const Eigen::Vector3d& color) {
    if (m_associatedCloud) {
        ccHObjectCaster::ToPointCloud(m_associatedCloud)->addEigenColor(color);
    }
}

Eigen::Vector3d ccMesh::getVertexColor(size_t index) const {
    if (!m_associatedCloud) {
        return Eigen::Vector3d();
    }

    return ccHObjectCaster::ToPointCloud(m_associatedCloud)
            ->getEigenColor(index);
}

void ccMesh::setVertexColors(const std::vector<Eigen::Vector3d>& colors) {
    if (m_associatedCloud && colors.size() == m_associatedCloud->size()) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        cloud->setEigenColors(colors);
    }
}

std::vector<Eigen::Vector3d> ccMesh::getVertexColors() const {
    if (!m_associatedCloud) {
        return std::vector<Eigen::Vector3d>();
    }

    return ccHObjectCaster::ToPointCloud(m_associatedCloud)->getEigenColors();
}

ColorsTableType* ccMesh::getVertexColorsPtr() {
    if (!m_associatedCloud) {
        return nullptr;
    }

    return ccHObjectCaster::ToPointCloud(m_associatedCloud)->rgbColors();
}

void ccMesh::addVertexColors(const std::vector<Eigen::Vector3d>& colors) {
    if (m_associatedCloud) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        if (cloud->reserveTheRGBTable()) {
            cloud->addEigenColors(colors);
        }
    }
}

unsigned int ccMesh::getVerticeSize() const {
    if (!getAssociatedCloud()) {
        return 0;
    }

    return getAssociatedCloud()->size();
}

std::vector<CCVector3>& ccMesh::getVerticesPtr() {
    if (!getAssociatedCloud()) {
        cloudViewer::utility::LogError(
                "[ccMesh] m_associatedCloud must be set before use!");
    }
    return ccHObjectCaster::ToPointCloud(getAssociatedCloud())->getPoints();
}

const std::vector<CCVector3>& ccMesh::getVertices() const {
    if (!getAssociatedCloud()) {
        cloudViewer::utility::LogError(
                "[ccMesh] m_associatedCloud must be set before use!");
    }

    return ccHObjectCaster::ToPointCloud(getAssociatedCloud())->getPoints();
}

void ccMesh::setEigenVertices(const std::vector<Eigen::Vector3d>& vertices) {
    if (m_associatedCloud && vertices.size() == m_associatedCloud->size()) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        cloud->setEigenPoints(vertices);
    }
}

std::vector<Eigen::Vector3d> ccMesh::getEigenVertices() const {
    if (!getAssociatedCloud()) {
        return std::vector<Eigen::Vector3d>();
    }

    return ccHObjectCaster::ToPointCloud(getAssociatedCloud())
            ->getEigenPoints();
}

void ccMesh::addEigenVertices(const std::vector<Eigen::Vector3d>& vertices) {
    if (m_associatedCloud) {
        ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(m_associatedCloud);
        if (cloud->reserveThePointsTable(
                    static_cast<unsigned>(cloud->size() + vertices.size()))) {
            cloud->addPoints(vertices);
        }
    }
}

void ccMesh::refreshBB() {
    if (!m_associatedCloud || m_bBox.isValid()) return;

    m_bBox.clear();

    size_t count = m_triVertIndexes->size();
    for (size_t i = 0; i < count; ++i) {
        const cloudViewer::VerticesIndexes& tri = m_triVertIndexes->at(i);
        assert(tri.i1 < m_associatedCloud->size() &&
               tri.i2 < m_associatedCloud->size() &&
               tri.i3 < m_associatedCloud->size());
        m_bBox.add(*m_associatedCloud->getPoint(tri.i1));
        m_bBox.add(*m_associatedCloud->getPoint(tri.i2));
        m_bBox.add(*m_associatedCloud->getPoint(tri.i3));
    }

    notifyGeometryUpdate();
}

void ccMesh::getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) {
    refreshBB();

    bbMin = m_bBox.minCorner();
    bbMax = m_bBox.maxCorner();
}

ccBBox ccMesh::getOwnBB(bool withGLFeatures /*=false*/) {
    refreshBB();

    return m_bBox;
}

const ccGLMatrix& ccMesh::getGLTransformationHistory() const {
    // DGM: it may happen that the vertices transformation history matrix is not
    // the same as the mesh (if applyGLTransformation is called directly on the
    // vertices). Therefore we prefer the cloud's by default.
    return m_associatedCloud ? m_associatedCloud->getGLTransformationHistory()
                             : m_glTransHistory;
}

// specific methods
void ccMesh::addTriangle(unsigned i1, unsigned i2, unsigned i3) {
    m_triVertIndexes->emplace_back(cloudViewer::VerticesIndexes(i1, i2, i3));
}

void ccMesh::addTriangle(const cloudViewer::VerticesIndexes& triangle) {
    m_triVertIndexes->emplace_back(triangle);
}

void ccMesh::setTriangle(size_t index, const Eigen::Vector3i& triangle) {
    if (index >= m_triVertIndexes->size()) {
        cloudViewer::utility::LogWarning(
                "[ccMesh::setTriangle] index out of range!");
        return;
    }
    m_triVertIndexes->at(index) =
            cloudViewer::VerticesIndexes(static_cast<unsigned>(triangle[0]),
                                         static_cast<unsigned>(triangle[1]),
                                         static_cast<unsigned>(triangle[2]));
}

Eigen::Vector3i ccMesh::getTriangle(size_t index) const {
    const cloudViewer::VerticesIndexes* tsi =
            getTriangleVertIndexes(static_cast<unsigned int>(index));

    return Eigen::Vector3i(tsi->i1, tsi->i2, tsi->i3);
}

void ccMesh::setTriangles(const std::vector<Eigen::Vector3i>& triangles) {
    resize(triangles.size());
    for (unsigned int i = 0; i < size(); ++i) {
        setTriangle(i, triangles[i]);
    }
}

std::vector<Eigen::Vector3i> ccMesh::getTriangles() const {
    if (!hasTriangles()) {
        cloudViewer::utility::LogWarning("[getTriangles] has no triangles!");
        return std::vector<Eigen::Vector3i>();
    }

    std::vector<Eigen::Vector3i> triangles(size());
    for (size_t i = 0; i < size(); ++i) {
        triangles[i] = getTriangle(i);
    }
    return triangles;
}

bool ccMesh::reserve(size_t n) {
    if (m_triNormalIndexes)
        if (!m_triNormalIndexes->reserveSafe(n)) return false;

    if (m_triMtlIndexes)
        if (!m_triMtlIndexes->reserveSafe(n)) return false;

    if (m_texCoordIndexes)
        if (!m_texCoordIndexes->reserveSafe(n)) return false;

    return m_triVertIndexes->reserveSafe(n);
}

bool ccMesh::resize(size_t n) {
    m_bBox.setValidity(false);
    notifyGeometryUpdate();

    if (m_triMtlIndexes) {
        static const int s_defaultMtlIndex = -1;
        if (!m_triMtlIndexes->resizeSafe(n, true, &s_defaultMtlIndex))
            return false;
    }

    if (m_texCoordIndexes) {
        static const Tuple3i s_defaultTexCoords(-1, -1, -1);
        if (!m_texCoordIndexes->resizeSafe(n, true, &s_defaultTexCoords))
            return false;
    }

    if (m_triNormalIndexes) {
        static const Tuple3i s_defaultNormIndexes(-1, -1, -1);
        if (!m_triNormalIndexes->resizeSafe(n, true, &s_defaultNormIndexes))
            return false;
    }

    return m_triVertIndexes->resizeSafe(n);
}

bool ccMesh::resizeAssociatedCloud(std::size_t n) {
    if (!m_associatedCloud) {
        cloudViewer::utility::LogWarning(
                "Must call createInternalCloud first!");
        return false;
    }
    ccPointCloud* baseVertices =
            ccHObjectCaster::ToPointCloud(m_associatedCloud);

    if (!baseVertices->resize(n)) {
        cloudViewer::utility::LogError("[resize] Not have enough memory! ");
        return false;
    }
    if (baseVertices->hasNormals() && !baseVertices->resizeTheNormsTable()) {
        cloudViewer::utility::LogError(
                "[resizeTheNormsTable] Not have enough memory! ");
        return false;
    }
    if (baseVertices->hasColors() && !baseVertices->resizeTheRGBTable()) {
        cloudViewer::utility::LogError(
                "[resizeTheRGBTable] Not have enough memory! ");
        return false;
    }
    return true;
}

bool ccMesh::reserveAssociatedCloud(std::size_t n,
                                    bool init_color,
                                    bool init_normal) {
    if (!m_associatedCloud) {
        cloudViewer::utility::LogWarning(
                "Must call createInternalCloud first!");
        return false;
    }
    ccPointCloud* baseVertices =
            ccHObjectCaster::ToPointCloud(m_associatedCloud);

    if (!baseVertices->reserveThePointsTable(n)) {
        cloudViewer::utility::LogError(
                "[reserveThePointsTable] Not have enough memory! ");
        return false;
    }
    if (init_normal && !baseVertices->reserveTheNormsTable()) {
        cloudViewer::utility::LogError(
                "[reserveTheNormsTable] Not have enough memory! ");
        return false;
    }
    if (init_color && !baseVertices->reserveTheRGBTable()) {
        cloudViewer::utility::LogError(
                "[reserveTheRGBTable] Not have enough memory! ");
        return false;
    }
    return true;
}

void ccMesh::shrinkVertexToFit() {
    if (m_associatedCloud) {
        ccHObjectCaster::ToPointCloud(m_associatedCloud)->shrinkToFit();
    }
}

void ccMesh::swapTriangles(unsigned index1, unsigned index2) {
    assert(std::max(index1, index2) < size());

    m_triVertIndexes->swap(index1, index2);
    if (m_triMtlIndexes) m_triMtlIndexes->swap(index1, index2);
    if (m_texCoordIndexes) m_texCoordIndexes->swap(index1, index2);
    if (m_triNormalIndexes) m_triNormalIndexes->swap(index1, index2);
}

void ccMesh::removeTriangles(size_t index) {
    if (index >= size()) {
        cloudViewer::utility::LogWarning(
                "[ccMesh::removeTriangles] index out of range!");
        return;
    }

    m_triVertIndexes->erase(m_triVertIndexes->begin() + index);
    if (m_triMtlIndexes)
        m_triMtlIndexes->erase(m_triMtlIndexes->begin() + index);
    if (m_texCoordIndexes)
        m_texCoordIndexes->erase(m_texCoordIndexes->begin() + index);
    if (m_triNormalIndexes)
        m_triNormalIndexes->erase(m_triNormalIndexes->begin() + index);
}

cloudViewer::VerticesIndexes* ccMesh::getTriangleVertIndexes(
        unsigned triangleIndex) {
    return &m_triVertIndexes->at(triangleIndex);
}

const cloudViewer::VerticesIndexes* ccMesh::getTriangleVertIndexes(
        unsigned triangleIndex) const {
    return &m_triVertIndexes->at(triangleIndex);
}

void ccMesh::getTriangleVertIndexes(size_t triangleIndex,
                                    Eigen::Vector3i& vertIndx) const {
    const cloudViewer::VerticesIndexes* tsi =
            getTriangleVertIndexes(static_cast<unsigned int>(triangleIndex));
    vertIndx(0) = tsi->i1;
    vertIndx(1) = tsi->i2;
    vertIndx(2) = tsi->i3;
}

cloudViewer::VerticesIndexes* ccMesh::getNextTriangleVertIndexes() {
    if (m_globalIterator < m_triVertIndexes->size()) {
        return getTriangleVertIndexes(m_globalIterator++);
    }

    return nullptr;
}

unsigned ccMesh::getUniqueIDForDisplay() const {
    if (m_parent && m_parent->getParent() &&
        m_parent->getParent()->isA(CV_TYPES::FACET)) {
        return m_parent->getParent()->getUniqueID();
    } else {
        return getUniqueID();
    }
}

void ccMesh::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (!m_associatedCloud) return;

    handleColorRamp(context);

    if (!ecvDisplayTools::GetMainWindow()) return;

    // 3D pass
    if (MACRO_Draw3D(context)) {
        // any triangle?
        size_t triNum = m_triVertIndexes->size();
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

        // display parameters
        glDrawParams glParams;
        getDrawingParameters(glParams);
        // no normals shading without light!
        if (!MACRO_LightIsEnabled(context)) glParams.showNorms = false;

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
        glParams.showNorms =
                showTriNormals ||
                (m_associatedCloud->hasNormals() && m_normalsDisplayed);

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
            // glParams.showSF --> we keep it only if SF 'NaN' values are hidden
            showTriNormals = false;
            applyMaterials = false;
            showTextures = false;
        }

        bool greyForNanScalarValues = true;
        // unsigned colorRampSteps = 0;
        ccColorScale::Shared colorScale(nullptr);

        if (glParams.showSF) {
            assert(m_associatedCloud->isA(CV_TYPES::POINT_CLOUD));
            ccPointCloud* cloud = static_cast<ccPointCloud*>(m_associatedCloud);

            greyForNanScalarValues = (cloud->getCurrentDisplayedScalarField() &&
                                      cloud->getCurrentDisplayedScalarField()
                                              ->areNaNValuesShownInGrey());
            if (greyForNanScalarValues && pushName) {
                // in picking mode, no need to take SF into account if we don't
                // hide any points!
                glParams.showSF = false;
            }
        }

        if (glParams.showColors) {
            if (isColorOverridden()) {
                context.defaultMeshColor = m_tempColor;
            } else {
                assert(m_associatedCloud->isA(CV_TYPES::POINT_CLOUD));
                context.defaultMeshColor =
                        static_cast<ccPointCloud*>(m_associatedCloud)
                                ->rgbColors()
                                ->getValue(0);
            }
        } else {
            context.defaultMeshColor = ecvColor::lightGrey;
        }

        context.drawParam = glParams;

        // vertices visibility
        const ccGenericPointCloud::VisibilityTableType& verticesVisibility =
                m_associatedCloud->getTheVisibilityArray();
        bool visFiltering =
                (verticesVisibility.size() >= m_associatedCloud->size());
        context.visFiltering = visFiltering;
        ecvDisplayTools::Draw(context, this);
    }
}

ccMesh* ccMesh::createNewMeshFromSelection(
        bool removeSelectedTriangles,
        std::vector<int>* newIndexesOfRemainingTriangles /*=nullptr*/,
        bool withChildEntities /*=false*/) {
    if (!m_associatedCloud) {
        return nullptr;
    }

    size_t triCount = size();

    // we always need a map of the new triangle indexes
    std::vector<int> triangleIndexMap;

    // we create a new mesh with the current selection
    ccMesh* newMesh = nullptr;
    {
        // create a 'reference' cloud if none was provided
        cloudViewer::ReferenceCloud rc(m_associatedCloud);

        // create vertices for the new mesh
        ccGenericPointCloud* newVertices =
                m_associatedCloud->createNewCloudFromVisibilitySelection(
                        false, nullptr, nullptr, true, &rc);
        if (!newVertices) {
            CVLog::Warning(
                    "[ccMesh::createNewMeshFromSelection] Failed to create "
                    "segmented mesh vertices! (not enough memory)");
            return nullptr;
        } else if (newVertices == m_associatedCloud) {
            // nothing to do
            return this;
        } else if (newVertices->size() == 0) {
            CVLog::Warning(
                    "[ccMesh::createNewMeshFromSelection] No visible point in "
                    "selection");
            delete newVertices;
            return nullptr;
        }
        assert(newVertices);

        assert(rc.size() !=
               0);  // otherwise 'newVertices->size() == 0' (see above)
        assert(rc.size() !=
               m_associatedCloud->size());  // in this case
                                            // createNewCloudFromVisibilitySelection
                                            // would have return
                                            // 'm_associatedCloud' itself

        cloudViewer::GenericIndexedMesh* selection =
                cloudViewer::ManualSegmentationTools::segmentMesh(
                        this, &rc, true, nullptr, newVertices, 0,
                        &triangleIndexMap);
        if (!selection) {
            CVLog::Warning(
                    "[ccMesh::createNewMeshFromSelection] Process failed: not "
                    "enough memory?");
            return nullptr;
        }

        newMesh = new ccMesh(selection, newVertices);

        delete selection;
        selection = nullptr;

        if (!newMesh) {
            delete newVertices;
            newVertices = nullptr;
            CVLog::Warning(
                    "[ccMesh::createNewMeshFromSelection] An error occurred: "
                    "not enough memory?");
            return nullptr;
        }
        newMesh->addChild(newVertices);
        newVertices->setEnabled(false);
    }
    assert(newMesh);

    // populate the new mesh
    {
        newMesh->setName(getName() + QString(".part"));

        // shall we add any advanced features?
        bool addFeatures = false;
        if (m_triNormals && m_triNormalIndexes)
            addFeatures |= newMesh->reservePerTriangleNormalIndexes();
        if (m_materials && m_triMtlIndexes)
            addFeatures |= newMesh->reservePerTriangleMtlIndexes();
        if (m_texCoords && m_texCoordIndexes)
            addFeatures |= newMesh->reservePerTriangleTexCoordIndexes();

        if (addFeatures) {
            // temporary structure for normal indexes mapping
            std::vector<int> newNormIndexes;
            NormsIndexesTableType* newTriNormals = nullptr;
            if (m_triNormals && m_triNormalIndexes) {
                assert(m_triNormalIndexes->size() == triCount);
                // create new 'minimal' subset
                newTriNormals = new NormsIndexesTableType();
                newTriNormals->link();
                try {
                    newNormIndexes.resize(m_triNormals->size(), -1);
                } catch (const std::bad_alloc&) {
                    CVLog::Warning(
                            "[ccMesh::createNewMeshFromSelection] Failed to "
                            "create new normals subset! (not enough memory)");
                    newMesh->removePerTriangleNormalIndexes();
                    newTriNormals->release();
                    newTriNormals = nullptr;
                }
            }

            // temporary structure for texture indexes mapping
            std::vector<int> newTexIndexes;
            TextureCoordsContainer* newTriTexIndexes = nullptr;
            if (m_texCoords && m_texCoordIndexes) {
                assert(m_texCoordIndexes->size() == triCount);
                // create new 'minimal' subset
                newTriTexIndexes = new TextureCoordsContainer();
                newTriTexIndexes->link();
                try {
                    newTexIndexes.resize(m_texCoords->size(), -1);
                } catch (const std::bad_alloc&) {
                    CVLog::Warning(
                            "[ccMesh::createNewMeshFromSelection] Failed to "
                            "create new texture indexes subset! (not enough "
                            "memory)");
                    newMesh->removePerTriangleTexCoordIndexes();
                    newTriTexIndexes->release();
                    newTriTexIndexes = nullptr;
                }
            }

            // temporary structure for material indexes mapping
            std::vector<int> newMatIndexes;
            ccMaterialSet* newMaterials = nullptr;
            if (m_materials && m_triMtlIndexes) {
                assert(m_triMtlIndexes->size() == triCount);
                // create new 'minimal' subset
                newMaterials = new ccMaterialSet(m_materials->getName() +
                                                 QString(".subset"));
                newMaterials->link();
                try {
                    newMatIndexes.resize(m_materials->size(), -1);
                } catch (const std::bad_alloc&) {
                    CVLog::Warning(
                            "[ccMesh::createNewMeshFromSelection] Failed to "
                            "create new material subset! (not enough memory)");
                    newMesh->removePerTriangleMtlIndexes();
                    newMaterials->release();
                    newMaterials = nullptr;
                    if (newTriTexIndexes)  // we can release texture coordinates
                                           // as well (as they depend on
                                           // materials!)
                    {
                        newMesh->removePerTriangleTexCoordIndexes();
                        newTriTexIndexes->release();
                        newTriTexIndexes = nullptr;
                        newTexIndexes.resize(0);
                    }
                }
            }

            for (size_t i = 0; i < triCount; ++i) {
                if (triangleIndexMap[i] >= 0)  // triangle should be copied over
                {
                    // import per-triangle normals?
                    if (newTriNormals) {
                        assert(m_triNormalIndexes);

                        // current triangle (compressed) normal indexes
                        const Tuple3i& triNormIndexes =
                                m_triNormalIndexes->getValue(i);

                        // for each triangle of this mesh, try to determine if
                        // its normals are already in use (otherwise add them to
                        //the new container and increase its index)
                        for (unsigned j = 0; j < 3; ++j) {
                            if (triNormIndexes.u[j] >= 0 &&
                                newNormIndexes[triNormIndexes.u[j]] < 0) {
                                if (newTriNormals->size() ==
                                            newTriNormals->capacity() &&
                                    !newTriNormals->reserveSafe(
                                            newTriNormals->size() +
                                            4096))  // auto expand
                                {
                                    CVLog::Warning(
                                            "[ccMesh::"
                                            "createNewMeshFromSelection] "
                                            "Failed to create new normals "
                                            "subset! (not enough memory)");
                                    newMesh->removePerTriangleNormalIndexes();
                                    newTriNormals->release();
                                    newTriNormals = nullptr;
                                    break;
                                }

                                // import old normal to new subset (create new
                                // index)
                                newNormIndexes[triNormIndexes.u[j]] =
                                        static_cast<int>(
                                                newTriNormals
                                                        ->size());  // new
                                                                    // element
                                                                    // index =
                                                                    // new size
                                                                    // - 1 = old
                                                                    // size!
                                newTriNormals->emplace_back(
                                        m_triNormals->getValue(
                                                triNormIndexes.u[j]));
                            }
                        }

                        if (newTriNormals)  // structure still exists?
                        {
                            newMesh->addTriangleNormalIndexes(
                                    triNormIndexes.u[0] < 0
                                            ? -1
                                            : newNormIndexes[triNormIndexes
                                                                     .u[0]],
                                    triNormIndexes.u[1] < 0
                                            ? -1
                                            : newNormIndexes[triNormIndexes
                                                                     .u[1]],
                                    triNormIndexes.u[2] < 0
                                            ? -1
                                            : newNormIndexes[triNormIndexes
                                                                     .u[2]]);
                        }
                    }

                    // import texture coordinates?
                    if (newTriTexIndexes) {
                        assert(m_texCoordIndexes);

                        // current triangle texture coordinates indexes
                        const Tuple3i& triTexIndexes =
                                m_texCoordIndexes->getValue(i);

                        // for each triangle of this mesh, try to determine if
                        // its textures coordinates are already in use
                        //(otherwise add them to the new container and increase
                        //its index)
                        for (unsigned j = 0; j < 3; ++j) {
                            if (triTexIndexes.u[j] >= 0 &&
                                newTexIndexes[triTexIndexes.u[j]] < 0) {
                                if (newTriTexIndexes->size() ==
                                            newTriTexIndexes->capacity() &&
                                    !newTriTexIndexes->reserveSafe(
                                            newTriTexIndexes->size() +
                                            4096))  // auto expand
                                {
                                    CVLog::Warning(
                                            "Failed to create new texture "
                                            "coordinates subset! (not enough "
                                            "memory)");
                                    newMesh->removePerTriangleTexCoordIndexes();
                                    newTriTexIndexes->release();
                                    newTriTexIndexes = nullptr;
                                    break;
                                }
                                // import old texture coordinate to new subset
                                // (create new index)
                                newTexIndexes[triTexIndexes.u[j]] = static_cast<
                                        int>(
                                        newTriTexIndexes
                                                ->size());  // new element index
                                                            // = new size - 1 =
                                                            // old size!
                                newTriTexIndexes->emplace_back(
                                        m_texCoords->getValue(
                                                triTexIndexes.u[j]));
                            }
                        }

                        if (newTriTexIndexes)  // structure still exists?
                        {
                            newMesh->addTriangleTexCoordIndexes(
                                    triTexIndexes.u[0] < 0
                                            ? -1
                                            : newTexIndexes[triTexIndexes.u[0]],
                                    triTexIndexes.u[1] < 0
                                            ? -1
                                            : newTexIndexes[triTexIndexes.u[1]],
                                    triTexIndexes.u[2] < 0
                                            ? -1
                                            : newTexIndexes[triTexIndexes
                                                                    .u[2]]);
                        }
                    }

                    // import materials?
                    if (newMaterials) {
                        assert(m_triMtlIndexes);

                        // current triangle material index
                        const int triMatIndex = m_triMtlIndexes->getValue(i);

                        // for each triangle of this mesh, try to determine if
                        // its material is already in use (otherwise add it to
                        //the new container and increase its index)
                        if (triMatIndex >= 0 &&
                            newMatIndexes[triMatIndex] < 0) {
                            // import old material to new subset (create new
                            // index)
                            newMatIndexes[triMatIndex] = static_cast<int>(
                                    newMaterials->size());  // new element index
                                                            // = new size - 1 =
                                                            // old size!
                            try {
                                newMaterials->emplace_back(
                                        m_materials->at(triMatIndex));
                            } catch (const std::bad_alloc&) {
                                CVLog::Warning(
                                        "[ccMesh::createNewMeshFromSelection] "
                                        "Failed to create new materials "
                                        "subset! (not enough memory)");
                                newMesh->removePerTriangleMtlIndexes();
                                newMaterials->release();
                                newMaterials = nullptr;
                            }
                        }

                        if (newMaterials)  // structure still exists?
                        {
                            newMesh->addTriangleMtlIndex(
                                    triMatIndex < 0
                                            ? -1
                                            : newMatIndexes[triMatIndex]);
                        }
                    }
                }
            }

            if (newTriNormals) {
                newTriNormals->resize(
                        newTriNormals->size());  // smaller so it should always
                                                 // be ok!
                newMesh->setTriNormsTable(newTriNormals);
                newTriNormals->release();
                newTriNormals = nullptr;
            }

            if (newTriTexIndexes) {
                newMesh->setTexCoordinatesTable(newTriTexIndexes);
                newTriTexIndexes->release();
                newTriTexIndexes = nullptr;
            }

            if (newMaterials) {
                newMesh->setMaterialSet(newMaterials);
                newMaterials->release();
                newMaterials = nullptr;
            }
        }

        newMesh->showColors(colorsShown());
        newMesh->showNormals(normalsShown());
        newMesh->showMaterials(materialsShown());
        newMesh->showSF(sfShown());
        newMesh->importParametersFrom(this);
    }

    // we must update eventual sub-meshes
    ccHObject::Container subMeshes;
    if (filterChildren(subMeshes, false, CV_TYPES::SUB_MESH) != 0) {
        // create index map
        try {
            ccSubMesh::IndexMap newRemainingTriangleIndexes;
            if (removeSelectedTriangles) {
                newRemainingTriangleIndexes.resize(
                        triCount, static_cast<unsigned>(triCount));

                unsigned newInvisibleIndex = 0;
                for (size_t i = 0; i < triCount; ++i) {
                    if (triangleIndexMap[i] <
                        0)  // triangle is not used in the new mesh, it will be
                            // kept in this one
                    {
                        newRemainingTriangleIndexes[i] = newInvisibleIndex++;
                    }
                }
            }

            for (size_t i = 0; i < subMeshes.size(); ++i) {
                ccSubMesh* subMesh = static_cast<ccSubMesh*>(subMeshes[i]);
                ccSubMesh* newSubMesh = subMesh->createNewSubMeshFromSelection(
                        removeSelectedTriangles, triangleIndexMap,
                        removeSelectedTriangles ? &newRemainingTriangleIndexes
                                                : nullptr);

                if (newSubMesh) {
                    if (newMesh) {
                        newSubMesh->setEnabled(subMesh->isEnabled());
                        newSubMesh->setVisible(subMesh->isVisible());
                        newSubMesh->setAssociatedMesh(newMesh);
                        newMesh->addChild(newSubMesh);
                    } else {
                        assert(false);
                        delete newSubMesh;
                        newSubMesh = nullptr;
                    }
                }

                if (subMesh->size() ==
                    0)  // no triangle left in current sub-mesh?
                {
                    removeChild(subMesh);
                    subMeshes[i] = nullptr;
                    subMesh = nullptr;
                }
            }
        } catch (const std::bad_alloc&) {
            CVLog::Warning("Not enough memory! Sub-meshes will be lost...");
            if (newMesh) {
                newMesh->setVisible(
                        true);  // force parent mesh visibility in this case!
            }

            for (size_t i = 0; i < subMeshes.size(); ++i) {
                removeChild(subMeshes[i]);
            }
        }
    }

    if (withChildEntities) {
        ccHObjectCaster::CloneChildren(this, newMesh, &triangleIndexMap);
    }

    // shall we remove the selected triangles from this mesh
    if (removeSelectedTriangles) {
        if (newIndexesOfRemainingTriangles) {
            if (newIndexesOfRemainingTriangles->empty()) {
                try {
                    newIndexesOfRemainingTriangles->resize(triCount);
                } catch (const std::bad_alloc&) {
                    CVLog::Warning(
                            "[ccMesh::createNewMeshFromSelection] Not enough "
                            "memory");
                    return nullptr;
                }
            } else if (newIndexesOfRemainingTriangles->size() != triCount) {
                CVLog::Warning(
                        "[ccMesh::createNewMeshFromSelection] Input 'new "
                        "indexes of reamining triangles' vector has a wrong "
                        "size");
                return nullptr;
            }
        }
        assert(!newIndexesOfRemainingTriangles ||
               newIndexesOfRemainingTriangles->size() == triCount);

        // we need to change the visibility status of some vertices that belong
        // to partially 'invisible' triangles
        auto& visArray = m_associatedCloud->getTheVisibilityArray();
        assert(visArray.size() == m_associatedCloud->size());

        size_t lastTri = 0;
        for (size_t i = 0; i < triCount; ++i) {
            if (triangleIndexMap[i] < 0)  // triangle is not used in the new
                                          // mesh, it will be kept in this one
            {
                const cloudViewer::VerticesIndexes& tsi = m_triVertIndexes->at(i);
                for (unsigned j = 0; j < 3; ++j) {
                    visArray[tsi.i[j]] = POINT_HIDDEN;
                }

                if (i != lastTri) {
                    m_triVertIndexes->setValue(lastTri, tsi);

                    if (m_triNormalIndexes)
                        m_triNormalIndexes->setValue(
                                lastTri, m_triNormalIndexes->getValue(i));
                    if (m_triMtlIndexes)
                        m_triMtlIndexes->setValue(lastTri,
                                                  m_triMtlIndexes->getValue(i));
                    if (m_texCoordIndexes)
                        m_texCoordIndexes->setValue(
                                lastTri, m_texCoordIndexes->getValue(i));
                }
                if (newIndexesOfRemainingTriangles) {
                    newIndexesOfRemainingTriangles->at(i) =
                            static_cast<int>(lastTri);
                }
                ++lastTri;
            } else if (newIndexesOfRemainingTriangles) {
                newIndexesOfRemainingTriangles->at(i) = -1;
            }
        }

        // update the mesh size
        resize(lastTri);
        triCount = size();

        std::vector<int> newIndexes;
        if (m_associatedCloud->removeVisiblePoints(nullptr, &newIndexes)) {
            // warning: from this point on, verticesVisibility is not valid
            // anymore!
            for (size_t i = 0; i < m_triVertIndexes->size(); ++i) {
                cloudViewer::VerticesIndexes& tsi = m_triVertIndexes->at(i);

                // update each vertex index
                for (int j = 0; j < 3; ++j) {
                    int oldVertexIndex = tsi.i[j];
                    assert(oldVertexIndex < newIndexes.size());
                    tsi.i[j] = newIndexes[oldVertexIndex];
                    assert(tsi.i[j] < m_associatedCloud->size());
                }
            }
        } else {
            CVLog::Warning(
                    "[ccMesh::createNewMeshFromSelection] Failed to remove "
                    "unused vertices");
        }

        notifyGeometryUpdate();

        // TODO: should we take care of the children here?
    } else if (newIndexesOfRemainingTriangles) {
        CVLog::Warning(
                "[ccMesh::createNewMeshFromSelection] A 'new indexes of "
                "reamining triangles' vector was provided while no triangle "
                "shall be removed");
    }

    m_associatedCloud->unallocateVisibilityArray();

    return newMesh;
}

void ccMesh::shiftTriangleIndexes(unsigned shift) {
    for (size_t i = 0; i < m_triVertIndexes->size(); ++i) {
        cloudViewer::VerticesIndexes& ti = m_triVertIndexes->at(i);
        ti.i1 += shift;
        ti.i2 += shift;
        ti.i3 += shift;
    }
}

void ccMesh::invertNormals() {
    // per-triangle normals
    if (m_triNormals) {
        invertPerTriangleNormals();
    }

    // per-vertex normals
    ccPointCloud* pc = dynamic_cast<ccPointCloud*>(m_associatedCloud);
    if (pc && pc->hasNormals()) {
        pc->invertNormals();
    }
}

void ccMesh::invertPerTriangleNormals() {
    if (m_triNormals) {
        for (CompressedNormType& n : *m_triNormals) {
            ccNormalCompressor::InvertNormal(n);
        }
    }
}

void ccMesh::flipTriangles() {
    for (cloudViewer::VerticesIndexes& ti : *m_triVertIndexes) {
        std::swap(ti.i2, ti.i3);
    }
}

/*********************************************************/
/**************    PER-TRIANGLE NORMALS    ***************/
/*********************************************************/

bool ccMesh::arePerTriangleNormalsEnabled() const {
    return m_triNormalIndexes && m_triNormalIndexes->isAllocated();
}

void ccMesh::removePerTriangleNormalIndexes() {
    if (m_triNormalIndexes) m_triNormalIndexes->release();
    m_triNormalIndexes = nullptr;
}

bool ccMesh::reservePerTriangleNormalIndexes() {
    assert(!m_triNormalIndexes);  // try to avoid doing this twice!
    if (!m_triNormalIndexes) {
        m_triNormalIndexes = new triangleNormalsIndexesSet();
        m_triNormalIndexes->link();
    }

    assert(m_triVertIndexes && m_triVertIndexes->isAllocated());

    return m_triNormalIndexes->reserveSafe(m_triVertIndexes->capacity());
}

void ccMesh::addTriangleNormalIndexes(int i1, int i2, int i3) {
    assert(m_triNormalIndexes && m_triNormalIndexes->isAllocated());
    m_triNormalIndexes->emplace_back(Tuple3i(i1, i2, i3));
}

void ccMesh::setTriangleNormalIndexes(unsigned triangleIndex,
                                      int i1,
                                      int i2,
                                      int i3) {
    assert(m_triNormalIndexes && m_triNormalIndexes->size() > triangleIndex);
    m_triNormalIndexes->setValue(triangleIndex, Tuple3i(i1, i2, i3));
}

void ccMesh::getTriangleNormalIndexes(unsigned triangleIndex,
                                      int& i1,
                                      int& i2,
                                      int& i3) const {
    if (m_triNormalIndexes && m_triNormalIndexes->size() > triangleIndex) {
        const Tuple3i& indexes = m_triNormalIndexes->getValue(triangleIndex);
        i1 = indexes.u[0];
        i2 = indexes.u[1];
        i3 = indexes.u[2];
    } else {
        i1 = i2 = i3 = -1;
    }
}

bool ccMesh::getTriangleNormals(unsigned triangleIndex,
                                CCVector3& Na,
                                CCVector3& Nb,
                                CCVector3& Nc) const {
    if (m_triNormals && m_triNormalIndexes &&
        m_triNormalIndexes->size() > triangleIndex) {
        const Tuple3i& indexes = m_triNormalIndexes->getValue(triangleIndex);
        if (indexes.u[0] >= 0)
            Na = ccNormalVectors::GetUniqueInstance()->getNormal(
                    m_triNormals->getValue(indexes.u[0]));
        else
            Na = CCVector3(0, 0, 0);
        if (indexes.u[1] >= 0)
            Nb = ccNormalVectors::GetUniqueInstance()->getNormal(
                    m_triNormals->getValue(indexes.u[1]));
        else
            Nb = CCVector3(0, 0, 0);
        if (indexes.u[2] >= 0)
            Nc = ccNormalVectors::GetUniqueInstance()->getNormal(
                    m_triNormals->getValue(indexes.u[2]));
        else
            Nc = CCVector3(0, 0, 0);

        return true;
    }

    return false;
}

bool ccMesh::getTriangleNormals(unsigned triangleIndex,
                                double Na[3],
                                double Nb[3],
                                double Nc[3]) const {
    if (m_triNormals && m_triNormalIndexes &&
        m_triNormalIndexes->size() > triangleIndex) {
        const Tuple3i& indexes = m_triNormalIndexes->getValue(triangleIndex);
        if (indexes.u[0] >= 0) {
            const PointCoordinateType* n1 =
                    ccNormalVectors::GetUniqueInstance()
                            ->getNormal(m_triNormals->getValue(indexes.u[0]))
                            .u;
            Na[0] = static_cast<double>(n1[0]);
            Na[1] = static_cast<double>(n1[1]);
            Na[2] = static_cast<double>(n1[2]);
        } else {
            Na[0] = 0.0;
            Na[1] = 0.0;
            Na[2] = 0.0;
        }
        if (indexes.u[1] >= 0) {
            const PointCoordinateType* n2 =
                    ccNormalVectors::GetUniqueInstance()
                            ->getNormal(m_triNormals->getValue(indexes.u[1]))
                            .u;
            Nb[0] = static_cast<double>(n2[0]);
            Nb[1] = static_cast<double>(n2[1]);
            Nb[2] = static_cast<double>(n2[2]);
        } else {
            Nb[0] = 0.0;
            Nb[1] = 0.0;
            Nb[2] = 0.0;
        }

        if (indexes.u[2] >= 0) {
            const PointCoordinateType* n3 =
                    ccNormalVectors::GetUniqueInstance()
                            ->getNormal(m_triNormals->getValue(indexes.u[2]))
                            .u;
            Nc[0] = static_cast<double>(n3[0]);
            Nc[1] = static_cast<double>(n3[1]);
            Nc[2] = static_cast<double>(n3[2]);
        } else {
            Nc[0] = 0.0;
            Nc[1] = 0.0;
            Nc[2] = 0.0;
        }

        return true;
    }
    return false;
}

bool ccMesh::getTriangleNormals(unsigned triangleIndex,
                                Eigen::Vector3d& Na,
                                Eigen::Vector3d& Nb,
                                Eigen::Vector3d& Nc) const {
    return getTriangleNormals(triangleIndex, Na.data(), Nb.data(), Nc.data());
}

std::vector<Eigen::Vector3d> ccMesh::getTriangleNormals() const {
    std::vector<Eigen::Vector3d> triangleNormals;
    triangleNormals.reserve(this->size());
    for (size_t i = 0; i < this->size(); i++) {
        triangleNormals.emplace_back(getTriangleNorm(i));
    }
    return triangleNormals;
}

std::vector<CCVector3*> ccMesh::getTriangleNormalsPtr() const {
    std::vector<CCVector3*> triNormals;
    for (size_t i = 0; i < this->size(); i++) {
        triNormals.push_back(&ccNormalVectors::GetUniqueInstance()->getNormal(
                m_triNormals->getValue(i)));
    }
    return triNormals;
}

Eigen::Vector3d ccMesh::getTriangleNorm(size_t index) const {
    if (index >= m_triNormals->size()) return Eigen::Vector3d(0.0, 0.0, 0.0);
    const PointCoordinateType* n1 =
            ccNormalVectors::GetUniqueInstance()
                    ->getNormal(m_triNormals->getValue(index))
                    .u;
    return Eigen::Vector3d(n1[0], n1[1], n1[2]);
}

bool ccMesh::setTriangleNorm(size_t index,
                             const Eigen::Vector3d& triangle_normal) {
    if (!hasTriNormals() || m_triNormals->size() <= index) {
        return false;
    }

    CompressedNormType nIndex = ccNormalVectors::GetNormIndex(
            CCVector3::fromArray(triangle_normal));
    setTriangleNormalIndexes(index, nIndex);
    return true;
}

bool ccMesh::setTriangleNormalIndexes(size_t triangleIndex,
                                      CompressedNormType value) {
    if (!hasTriNormals() || m_triNormals->size() <= triangleIndex) {
        return false;
    }

    m_triNormals->setValue(triangleIndex, value);
    return true;
}

CompressedNormType ccMesh::getTriangleNormalIndexes(size_t triangleIndex) {
    if (!hasTriNormals() || m_triNormals->size() <= triangleIndex) {
        assert(false);
    }

    return m_triNormals->getValue(triangleIndex);
}

bool ccMesh::addTriangleNorm(const CCVector3& N) {
    if (!arePerTriangleNormalsEnabled()) {
        NormsIndexesTableType* normsTable = new NormsIndexesTableType();
        if (!reservePerTriangleNormalIndexes()) {
            normsTable->release();
            CVLog::Warning("[ccMesh::addTriangleNorm] Not enough memory!");
            return false;
        }
        setTriNormsTable(normsTable);
    }

    CompressedNormType nIndex = ccNormalVectors::GetNormIndex(N.u);
    m_triNormals->push_back(nIndex);
    int normalIndex = static_cast<int>(m_triNormals->size() - 1);
    addTriangleNormalIndexes(normalIndex, normalIndex, normalIndex);
    return true;
}

bool ccMesh::addTriangleNorm(const Eigen::Vector3d& N) {
    return addTriangleNorm(CCVector3::fromArray(N));
}

std::vector<Eigen::Vector3d> ccMesh::getTriangleNorms() const {
    if (!hasTriNormals()) {
        cloudViewer::utility::LogWarning(
                "[getTriangleNorms] has no triangle normals!");
        return std::vector<Eigen::Vector3d>();
    }

    std::vector<Eigen::Vector3d> triangle_normals(m_triNormals->size());
    for (size_t i = 0; i < m_triNormals->size(); ++i) {
        triangle_normals[i] = getTriangleNorm(i);
    }

    return triangle_normals;
}

bool ccMesh::setTriangleNorms(
        const std::vector<Eigen::Vector3d>& triangle_normals) {
    bool success = true;
    if (resize(triangle_normals.size()) && m_triNormals &&
        m_triNormals->resizeSafe(triangle_normals.size())) {
        for (size_t i = 0; i < triangle_normals.size(); ++i) {
            if (!setTriangleNorm(i, triangle_normals[i])) {
                cloudViewer::utility::LogWarning(
                        "[ccMesh::addTriangleNorms] add triangle normals "
                        "failed!");
                success = false;
                break;
            }
        }
    } else {
        success = false;
    }

    return success;
}

bool ccMesh::addTriangleNorms(
        const std::vector<Eigen::Vector3d>& triangle_normals) {
    bool success = true;
    if (reserve(size() + triangle_normals.size())) {
        for (const auto& normal : triangle_normals) {
            if (!addTriangleNorm(normal)) {
                cloudViewer::utility::LogWarning(
                        "[ccMesh::addTriangleNorms] add triangle normals "
                        "failed!");
                success = false;
                break;
            }
        }
    }

    return success;
}

bool ccMesh::hasTriNormals() const {
    return m_triNormals && m_triNormals->isAllocated() && m_triNormalIndexes &&
           (m_triNormalIndexes->size() == m_triVertIndexes->size());
}

/*********************************************************/
/************    PER-TRIANGLE TEX COORDS    **************/
/*********************************************************/

void ccMesh::setTexCoordinatesTable(TextureCoordsContainer* texCoordsTable,
                                    bool autoReleaseOldTable /*=true*/) {
    if (m_texCoords == texCoordsTable) return;

    if (m_texCoords && autoReleaseOldTable) {
        int childIndex = getChildIndex(m_texCoords);
        m_texCoords->release();
        m_texCoords = nullptr;
        if (childIndex >= 0) removeChild(childIndex);
    }

    m_texCoords = texCoordsTable;
    if (m_texCoords) {
        m_texCoords->link();
        int childIndex = getChildIndex(m_texCoords);
        if (childIndex < 0) addChild(m_texCoords);
    } else {
        removePerTriangleTexCoordIndexes();  // auto-remove per-triangle indexes
                                             // (we don't need them anymore)
    }
}

void ccMesh::getTriangleTexCoordinates(unsigned triIndex,
                                       TexCoords2D*& tx1,
                                       TexCoords2D*& tx2,
                                       TexCoords2D*& tx3) const {
    if (m_texCoords && m_texCoordIndexes) {
        const Tuple3i& txInd = m_texCoordIndexes->getValue(triIndex);
        tx1 = (txInd.u[0] >= 0 ? &m_texCoords->getValue(txInd.u[0]) : nullptr);
        tx2 = (txInd.u[1] >= 0 ? &m_texCoords->getValue(txInd.u[1]) : nullptr);
        tx3 = (txInd.u[2] >= 0 ? &m_texCoords->getValue(txInd.u[2]) : nullptr);
    } else {
        tx1 = tx2 = tx3;
    }
}

void ccMesh::getTexCoordinates(unsigned index, TexCoords2D*& tx) const {
    if (m_texCoords && m_texCoords->size() > index) {
        tx = &m_texCoords->getValue(index);
    }
}

bool ccMesh::reservePerTriangleTexCoordIndexes() {
    assert(!m_texCoordIndexes);  // try to avoid doing this twice!
    if (!m_texCoordIndexes) {
        m_texCoordIndexes = new triangleTexCoordIndexesSet();
        m_texCoordIndexes->link();
    }

    assert(m_triVertIndexes && m_triVertIndexes->isAllocated());

    return m_texCoordIndexes->reserveSafe(m_triVertIndexes->capacity());
}

void ccMesh::removePerTriangleTexCoordIndexes() {
    triangleTexCoordIndexesSet* texCoordIndexes = m_texCoordIndexes;
    m_texCoordIndexes = nullptr;

    if (texCoordIndexes) texCoordIndexes->release();
}

void ccMesh::addTriangleTexCoordIndexes(int i1, int i2, int i3) {
    assert(m_texCoordIndexes && m_texCoordIndexes->isAllocated());
    m_texCoordIndexes->emplace_back(Tuple3i(i1, i2, i3));
}

void ccMesh::setTriangleTexCoordIndexes(unsigned triangleIndex,
                                        int i1,
                                        int i2,
                                        int i3) {
    assert(m_texCoordIndexes && m_texCoordIndexes->size() > triangleIndex);
    m_texCoordIndexes->setValue(triangleIndex, Tuple3i(i1, i2, i3));
}

void ccMesh::getTriangleTexCoordinatesIndexes(unsigned triangleIndex,
                                              int& i1,
                                              int& i2,
                                              int& i3) const {
    assert(m_texCoordIndexes && m_texCoordIndexes->size() > triangleIndex);

    const Tuple3i& tci = m_texCoordIndexes->getValue(triangleIndex);
    i1 = tci.u[0];
    i2 = tci.u[1];
    i3 = tci.u[2];
}

bool ccMesh::hasTextures() const {
    return hasMaterials() && m_texCoords && m_texCoords->isAllocated() &&
           m_texCoordIndexes &&
           (m_texCoordIndexes->size() == m_triVertIndexes->size());
}

/*********************************************************/
/**************    PER-TRIANGLE MATERIALS    *************/
/*********************************************************/

bool ccMesh::hasMaterials() const {
    return m_materials && !m_materials->empty() && m_triMtlIndexes &&
           (m_triMtlIndexes->size() == m_triVertIndexes->size());
}

void ccMesh::setTriangleMtlIndexesTable(
        triangleMaterialIndexesSet* matIndexesTable,
        bool autoReleaseOldTable /*=true*/) {
    if (m_triMtlIndexes == matIndexesTable) return;

    if (m_triMtlIndexes && autoReleaseOldTable) {
        m_triMtlIndexes->release();
        m_triMtlIndexes = nullptr;
    }

    m_triMtlIndexes = matIndexesTable;
    if (m_triMtlIndexes) {
        m_triMtlIndexes->link();
    }
}

bool ccMesh::reservePerTriangleMtlIndexes() {
    assert(!m_triMtlIndexes);  // try to avoid doing this twice!
    if (!m_triMtlIndexes) {
        m_triMtlIndexes = new triangleMaterialIndexesSet();
        m_triMtlIndexes->link();
    }

    assert(m_triVertIndexes && m_triVertIndexes->isAllocated());

    return m_triMtlIndexes->reserveSafe(m_triVertIndexes->capacity());
}

void ccMesh::removePerTriangleMtlIndexes() {
    if (m_triMtlIndexes) m_triMtlIndexes->release();
    m_triMtlIndexes = nullptr;
}

void ccMesh::addTriangleMtlIndex(int mtlIndex) {
    assert(m_triMtlIndexes && m_triMtlIndexes->isAllocated());
    m_triMtlIndexes->emplace_back(mtlIndex);
}

void ccMesh::setTriangleMtlIndex(unsigned triangleIndex, int mtlIndex) {
    assert(m_triMtlIndexes && m_triMtlIndexes->size() > triangleIndex);
    m_triMtlIndexes->setValue(triangleIndex, mtlIndex);
}

int ccMesh::getTriangleMtlIndex(unsigned triangleIndex) const {
    assert(m_triMtlIndexes && m_triMtlIndexes->size() > triangleIndex);
    return m_triMtlIndexes->at(triangleIndex);
}

bool ccMesh::toFile_MeOnly(QFile& out) const {
    if (!ccGenericMesh::toFile_MeOnly(out)) return false;

    // we can't save the associated cloud here (as it may be shared by multiple
    // meshes) so instead we save it's unique ID (dataVersion>=20) WARNING: the
    // cloud must be saved in the same BIN file! (responsibility of the caller)
    uint32_t vertUniqueID =
            (m_associatedCloud
                     ? static_cast<uint32_t>(m_associatedCloud->getUniqueID())
                     : 0);
    if (out.write((const char*)&vertUniqueID, 4) < 0) return WriteError();

    // per-triangle normals array (dataVersion>=20)
    {
        // we can't save the normals array here (as it may be shared by multiple
        // meshes) so instead we save it's unique ID (dataVersion>=20) WARNING:
        // the normals array must be saved in the same BIN file! (responsibility
        // of the caller)
        uint32_t normArrayID =
                (m_triNormals && m_triNormals->isAllocated()
                         ? static_cast<uint32_t>(m_triNormals->getUniqueID())
                         : 0);
        if (out.write((const char*)&normArrayID, 4) < 0) return WriteError();
    }

    // texture coordinates array (dataVersion>=20)
    {
        // we can't save the texture coordinates array here (as it may be shared
        // by multiple meshes) so instead we save it's unique ID
        // (dataVersion>=20) WARNING: the texture coordinates array must be
        // saved in the same BIN file! (responsibility of the caller)
        uint32_t texCoordArrayID =
                (m_texCoords && m_texCoords->isAllocated()
                         ? static_cast<uint32_t>(m_texCoords->getUniqueID())
                         : 0);
        if (out.write((const char*)&texCoordArrayID, 4) < 0)
            return WriteError();
    }

    // materials
    {
        // we can't save the material set here (as it may be shared by multiple
        // meshes) so instead we save it's unique ID (dataVersion>=20) WARNING:
        // the material set must be saved in the same BIN file! (responsibility
        // of the caller)
        uint32_t matSetID =
                (m_materials ? static_cast<uint32_t>(m_materials->getUniqueID())
                             : 0);
        if (out.write((const char*)&matSetID, 4) < 0) return WriteError();
    }

    // triangles indexes (dataVersion>=20)
    if (!m_triVertIndexes)
        return CVLog::Error(
                "Internal error: mesh has no triangles array! (not enough "
                "memory?)");
    if (!ccSerializationHelper::GenericArrayToFile<cloudViewer::VerticesIndexes,
                                                   3, unsigned>(
                *m_triVertIndexes, out))
        return false;

    // per-triangle materials (dataVersion>=20))
    bool hasTriMtlIndexes = hasPerTriangleMtlIndexes();
    if (out.write((const char*)&hasTriMtlIndexes, sizeof(bool)) < 0)
        return WriteError();
    if (hasTriMtlIndexes) {
        assert(m_triMtlIndexes);
        if (!ccSerializationHelper::GenericArrayToFile<int, 1, int>(
                    *m_triMtlIndexes, out))
            return false;
    }

    // per-triangle texture coordinates indexes (dataVersion>=20))
    bool hasTexCoordIndexes = hasPerTriangleTexCoordIndexes();
    if (out.write((const char*)&hasTexCoordIndexes, sizeof(bool)) < 0)
        return WriteError();
    if (hasTexCoordIndexes) {
        assert(m_texCoordIndexes);
        if (!ccSerializationHelper::GenericArrayToFile<Tuple3i, 3, int>(
                    *m_texCoordIndexes, out))
            return false;
    }

    // per-triangle normals  indexes (dataVersion>=20))
    bool hasTriNormalIndexes =
            (m_triNormalIndexes && m_triNormalIndexes->isAllocated());
    if (out.write((const char*)&hasTriNormalIndexes, sizeof(bool)) < 0)
        return WriteError();
    if (hasTriNormalIndexes) {
        assert(m_triNormalIndexes);
        if (!ccSerializationHelper::GenericArrayToFile<Tuple3i, 3, int>(
                    *m_triNormalIndexes, out))
            return false;
    }

    return true;
}

bool ccMesh::fromFile_MeOnly(QFile& in,
                             short dataVersion,
                             int flags,
                             LoadedIDMap& oldToNewIDMap) {
    if (!ccGenericMesh::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    // as the associated cloud (=vertices) can't be saved directly (as it may be
    // shared by multiple meshes) we only store its unique ID (dataVersion>=20)
    // --> we hope we will find it at loading time (i.e. this is the
    // responsibility of the caller to make sure that all dependencies are saved
    // together)
    uint32_t vertUniqueID = 0;
    if (in.read((char*)&vertUniqueID, 4) < 0) return ReadError();
    //[DIRTY] WARNING: temporarily, we set the vertices unique ID in the
    //'m_associatedCloud' pointer!!!
    *(uint32_t*)(&m_associatedCloud) = vertUniqueID;

    // per-triangle normals array (dataVersion>=20)
    {
        // as the associated normals array can't be saved directly (as it may be
        // shared by multiple meshes) we only store its unique ID
        // (dataVersion>=20) --> we hope we will find it at loading time (i.e.
        // this is the responsibility of the caller to make sure that all
        // dependencies are saved together)
        uint32_t normArrayID = 0;
        if (in.read((char*)&normArrayID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the array unique ID in the
        //'m_triNormals' pointer!!!
        *(uint32_t*)(&m_triNormals) = normArrayID;
    }

    // texture coordinates array (dataVersion>=20)
    {
        // as the associated texture coordinates array can't be saved directly
        // (as it may be shared by multiple meshes) we only store its unique ID
        // (dataVersion>=20) --> we hope we will find it at loading time (i.e.
        // this is the responsibility of the caller to make sure that all
        // dependencies are saved together)
        uint32_t texCoordArrayID = 0;
        if (in.read((char*)&texCoordArrayID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the array unique ID in the
        //'m_texCoords' pointer!!!
        *(uint32_t*)(&m_texCoords) = texCoordArrayID;
    }

    // materials
    {
        // as the associated materials can't be saved directly (as it may be
        // shared by multiple meshes) we only store its unique ID
        // (dataVersion>=20) --> we hope we will find it at loading time (i.e.
        // this is the responsibility of the caller to make sure that all
        // dependencies are saved together)
        uint32_t matSetID = 0;
        if (in.read((char*)&matSetID, 4) < 0) return ReadError();
        //[DIRTY] WARNING: temporarily, we set the array unique ID in the
        //'m_materials' pointer!!!
        *(uint32_t*)(&m_materials) = matSetID;
    }

    // triangles indexes (dataVersion>=20)
    if (!m_triVertIndexes) return false;
    if (!ccSerializationHelper::GenericArrayFromFile<
                cloudViewer::VerticesIndexes, 3, unsigned>(*m_triVertIndexes,
                                                           in, dataVersion))
        return false;

    // per-triangle materials (dataVersion>=20))
    bool hasTriMtlIndexes = false;
    if (in.read((char*)&hasTriMtlIndexes, sizeof(bool)) < 0) return ReadError();
    if (hasTriMtlIndexes) {
        if (!m_triMtlIndexes) {
            m_triMtlIndexes = new triangleMaterialIndexesSet();
            m_triMtlIndexes->link();
        }
        if (!ccSerializationHelper::GenericArrayFromFile<int, 1, int>(
                    *m_triMtlIndexes, in, dataVersion)) {
            m_triMtlIndexes->release();
            m_triMtlIndexes = nullptr;
            return false;
        }
    }

    // per-triangle texture coordinates indexes (dataVersion>=20))
    bool hasTexCoordIndexes = false;
    if (in.read((char*)&hasTexCoordIndexes, sizeof(bool)) < 0)
        return ReadError();
    if (hasTexCoordIndexes) {
        if (!m_texCoordIndexes) {
            m_texCoordIndexes = new triangleTexCoordIndexesSet();
            m_texCoordIndexes->link();
        }
        if (!ccSerializationHelper::GenericArrayFromFile<Tuple3i, 3, int>(
                    *m_texCoordIndexes, in, dataVersion)) {
            m_texCoordIndexes->release();
            m_texCoordIndexes = nullptr;
            return false;
        }
    }

    //'materials shown' state (dataVersion>=20 && dataVersion<29))
    if (dataVersion < 29) {
        bool materialsShown = false;
        if (in.read((char*)&materialsShown, sizeof(bool)) < 0)
            return ReadError();
        showMaterials(materialsShown);
    }

    // per-triangle normals  indexes (dataVersion>=20))
    bool hasTriNormalIndexes = false;
    if (in.read((char*)&hasTriNormalIndexes, sizeof(bool)) < 0)
        return ReadError();
    if (hasTriNormalIndexes) {
        if (!m_triNormalIndexes) {
            m_triNormalIndexes = new triangleNormalsIndexesSet();
            m_triNormalIndexes->link();
        }
        assert(m_triNormalIndexes);
        if (!ccSerializationHelper::GenericArrayFromFile<Tuple3i, 3, int>(
                    *m_triNormalIndexes, in, dataVersion)) {
            removePerTriangleNormalIndexes();
            return false;
        }
    }

    if (dataVersion < 29) {
        //'per-triangle normals shown' state (dataVersion>=20 &&
        // dataVersion<29))
        bool triNormsShown = false;
        if (in.read((char*)&triNormsShown, sizeof(bool)) < 0)
            return ReadError();
        showTriNorms(triNormsShown);

        //'polygon stippling' state (dataVersion>=20 && dataVersion<29))
        bool stippling = false;
        if (in.read((char*)&stippling, sizeof(bool)) < 0) return ReadError();
        enableStippling(stippling);
    }

    notifyGeometryUpdate();

    return true;
}

void ccMesh::computeInterpolationWeights(unsigned triIndex,
                                         const CCVector3& P,
                                         CCVector3d& weights) const {
    assert(triIndex < m_triVertIndexes->size());

    const cloudViewer::VerticesIndexes& tri = m_triVertIndexes->at(triIndex);
    return computeInterpolationWeights(tri, P, weights);
}

void ccMesh::computeInterpolationWeights(
        const cloudViewer::VerticesIndexes& vertIndexes,
        const CCVector3& P,
        CCVector3d& weights) const {
    const CCVector3* A = m_associatedCloud->getPoint(vertIndexes.i1);
    const CCVector3* B = m_associatedCloud->getPoint(vertIndexes.i2);
    const CCVector3* C = m_associatedCloud->getPoint(vertIndexes.i3);

    // barcyentric intepolation weights
    weights.x = sqrt(((P - *B).cross(*C - *B)).norm2d()) /*/2*/;
    weights.y = sqrt(((P - *C).cross(*A - *C)).norm2d()) /*/2*/;
    weights.z = sqrt(((P - *A).cross(*B - *A)).norm2d()) /*/2*/;

    // normalize weights
    double sum = weights.x + weights.y + weights.z;
    weights /= sum;
}

bool ccMesh::interpolateNormals(unsigned triIndex,
                                const CCVector3& P,
                                CCVector3& N) {
    assert(triIndex < size());

    if (!hasNormals()) return false;

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triIndex);

    return interpolateNormals(
            tri, P, N,
            hasTriNormals() ? &m_triNormalIndexes->at(triIndex) : nullptr);
}

bool ccMesh::interpolateNormals(const cloudViewer::VerticesIndexes& vertIndexes,
                                const CCVector3d& w,
                                CCVector3& N,
                                const Tuple3i* triNormIndexes /*=0*/) {
    CCVector3d Nd(0, 0, 0);
    {
        if (!triNormIndexes || triNormIndexes->u[0] >= 0) {
            const CCVector3& N1 =
                    triNormIndexes
                            ? ccNormalVectors::GetNormal(m_triNormals->getValue(
                                      triNormIndexes->u[0]))
                            : m_associatedCloud->getPointNormal(vertIndexes.i1);
            Nd += N1.toDouble() * w.u[0];
        }

        if (!triNormIndexes || triNormIndexes->u[1] >= 0) {
            const CCVector3& N2 =
                    triNormIndexes
                            ? ccNormalVectors::GetNormal(m_triNormals->getValue(
                                      triNormIndexes->u[1]))
                            : m_associatedCloud->getPointNormal(vertIndexes.i2);
            Nd += N2.toDouble() * w.u[1];
        }

        if (!triNormIndexes || triNormIndexes->u[2] >= 0) {
            const CCVector3& N3 =
                    triNormIndexes
                            ? ccNormalVectors::GetNormal(m_triNormals->getValue(
                                      triNormIndexes->u[2]))
                            : m_associatedCloud->getPointNormal(vertIndexes.i3);
            Nd += N3.toDouble() * w.u[2];
        }
        Nd.normalize();
    }

    N = Nd.toPC();

    return true;
}

bool ccMesh::interpolateNormalsBC(unsigned triIndex,
                                  const CCVector3d& w,
                                  CCVector3& N) {
    assert(triIndex < size());

    if (!hasNormals()) return false;

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triIndex);

    return interpolateNormals(
            tri, w, N,
            hasTriNormals() ? &m_triNormalIndexes->at(triIndex) : nullptr);
}

bool ccMesh::interpolateColors(unsigned triIndex,
                               const CCVector3& P,
                               ecvColor::Rgb& C) {
    assert(triIndex < size());

    if (!hasColors()) return false;

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triIndex);

    return interpolateColors(tri, P, C);
}

bool ccMesh::interpolateColors(const cloudViewer::VerticesIndexes& vertIndexes,
                               const CCVector3& P,
                               ecvColor::Rgb& rgb) {
    // intepolation weights
    CCVector3d w;
    computeInterpolationWeights(vertIndexes, P, w);

    const ecvColor::Rgb& C1 = m_associatedCloud->getPointColor(vertIndexes.i1);
    const ecvColor::Rgb& C2 = m_associatedCloud->getPointColor(vertIndexes.i2);
    const ecvColor::Rgb& C3 = m_associatedCloud->getPointColor(vertIndexes.i3);

    rgb.r = static_cast<ColorCompType>(
            floor(C1.r * w.u[0] + C2.r * w.u[1] + C3.r * w.u[2]));
    rgb.g = static_cast<ColorCompType>(
            floor(C1.g * w.u[0] + C2.g * w.u[1] + C3.g * w.u[2]));
    rgb.b = static_cast<ColorCompType>(
            floor(C1.b * w.u[0] + C2.b * w.u[1] + C3.b * w.u[2]));

    return true;
}

bool ccMesh::getVertexColorFromMaterial(unsigned triIndex,
                                        unsigned char vertIndex,
                                        ecvColor::Rgb& rgb,
                                        bool returnColorIfNoTexture) {
    assert(triIndex < size());

    assert(vertIndex < 3);
    if (vertIndex > 2) {
        CVLog::Error(
                "[ccMesh::getVertexColorFromMaterial] Internal error: invalid "
                "vertex index!");
        return false;
    }

    int matIndex = -1;

    if (hasMaterials()) {
        assert(m_materials);
        matIndex = m_triMtlIndexes->getValue(triIndex);
        assert(matIndex < static_cast<int>(m_materials->size()));
    }

    const cloudViewer::VerticesIndexes& tri =
            m_triVertIndexes->getValue(triIndex);

    // do we need to change material?
    bool foundMaterial = false;
    if (matIndex >= 0) {
        ccMaterial::CShared material = (*m_materials)[matIndex];
        if (material->hasTexture()) {
            assert(m_texCoords && m_texCoordIndexes);
            const Tuple3i& txInd = m_texCoordIndexes->getValue(triIndex);
            const TexCoords2D* T =
                    (txInd.u[vertIndex] >= 0
                             ? &m_texCoords->getValue(txInd.u[vertIndex])
                             : nullptr);
            if (T) {
                // get the texture coordinates between 0 and 1
                float temp;
                float tx = std::modf(T->tx, &temp);
                if (tx < 0) tx = 1.0f + tx;
                float ty = std::modf(T->ty, &temp);
                if (ty < 0) ty = 1.0f + ty;

                // get color from texture image
                const QImage texture = material->getTexture();
                int xPix =
                        std::min(static_cast<int>(floor(tx * texture.width())),
                                 texture.width() - 1);
                int yPix =
                        std::min(static_cast<int>(floor(ty * texture.height())),
                                 texture.height() - 1);

                QRgb pixel = texture.pixel(xPix, yPix);

                rgb = ecvColor::FromQRgb(pixel);
                foundMaterial = true;
            }
        } else {
            const ecvColor::Rgbaf& diffuse = material->getDiffuseFront();
            rgb = ecvColor::FromRgbafToRgb(diffuse);

            foundMaterial = true;
        }
    }

    if (!foundMaterial && returnColorIfNoTexture && hasColors()) {
        rgb = ecvColor::Rgb(m_associatedCloud->getPointColor(tri.i[vertIndex]));
        foundMaterial = true;
    }

    return foundMaterial;
}

bool ccMesh::getColorFromMaterial(unsigned triIndex,
                                  const CCVector3& P,
                                  ecvColor::Rgb& rgb,
                                  bool interpolateColorIfNoTexture) {
    assert(triIndex < size());

    int matIndex = -1;

    if (hasMaterials()) {
        assert(m_materials);
        matIndex = m_triMtlIndexes->getValue(triIndex);
        assert(matIndex < static_cast<int>(m_materials->size()));
    }

    // do we need to change material?
    if (matIndex < 0) {
        if (interpolateColorIfNoTexture)
            return interpolateColors(triIndex, P, rgb);
        return false;
    }

    ccMaterial::CShared material = (*m_materials)[matIndex];

    if (!material->hasTexture()) {
        const ecvColor::Rgbaf& diffuse = material->getDiffuseFront();
        rgb.r = static_cast<ColorCompType>(diffuse.r * ecvColor::MAX);
        rgb.g = static_cast<ColorCompType>(diffuse.g * ecvColor::MAX);
        rgb.b = static_cast<ColorCompType>(diffuse.b * ecvColor::MAX);
        return true;
    }

    assert(m_texCoords && m_texCoordIndexes);
    const Tuple3i& txInd = m_texCoordIndexes->getValue(triIndex);
    const TexCoords2D* T1 =
            (txInd.u[0] >= 0 ? &m_texCoords->getValue(txInd.u[0]) : nullptr);
    const TexCoords2D* T2 =
            (txInd.u[1] >= 0 ? &m_texCoords->getValue(txInd.u[1]) : nullptr);
    const TexCoords2D* T3 =
            (txInd.u[2] >= 0 ? &m_texCoords->getValue(txInd.u[2]) : nullptr);

    // intepolation weights
    CCVector3d w;
    computeInterpolationWeights(triIndex, P, w);

    if ((!T1 && w.u[0] > ZERO_TOLERANCE_D) ||
        (!T2 && w.u[1] > ZERO_TOLERANCE_D) ||
        (!T3 && w.u[2] > ZERO_TOLERANCE_D)) {
        // assert(false);
        if (interpolateColorIfNoTexture)
            return interpolateColors(triIndex, P, rgb);
        return false;
    }

    double x = (T1 ? T1->tx * w.u[0] : 0.0) + (T2 ? T2->tx * w.u[1] : 0.0) +
               (T3 ? T3->tx * w.u[2] : 0.0);
    double y = (T1 ? T1->ty * w.u[0] : 0.0) + (T2 ? T2->ty * w.u[1] : 0.0) +
               (T3 ? T3->ty * w.u[2] : 0.0);

    // DGM: we mut handle texture coordinates below 0 or above 1 (i.e.
    // repetition) if (x < 0 || x > 1.0 || y < 0 || y > 1.0)
    if (x > 1.0) {
        double xFrac, xInt;
        xFrac = std::modf(x, &xInt);
        x = xFrac;
    } else if (x < 0.0) {
        double xFrac, xInt;
        xFrac = std::modf(x, &xInt);
        x = 1.0 + xFrac;
    }

    // same thing for y
    if (y > 1.0) {
        double yFrac, yInt;
        yFrac = std::modf(y, &yInt);
        y = yFrac;
    } else if (y < 0.0) {
        double yFrac, yInt;
        yFrac = std::modf(y, &yInt);
        y = 1.0 + yFrac;
    }

    // get color from texture image
    {
        const QImage texture = material->getTexture();
        int xPix = std::min(static_cast<int>(floor(x * texture.width())),
                            texture.width() - 1);
        int yPix = std::min(static_cast<int>(floor(y * texture.height())),
                            texture.height() - 1);

        QRgb pixel = texture.pixel(xPix, yPix);

        const ecvColor::Rgbaf& diffuse = material->getDiffuseFront();
        rgb.r = static_cast<ColorCompType>(diffuse.r * qRed(pixel));
        rgb.g = static_cast<ColorCompType>(diffuse.g * qGreen(pixel));
        rgb.b = static_cast<ColorCompType>(diffuse.b * qBlue(pixel));
    }

    return true;
}

// we use as many static variables as we can to limit the size of the heap used
// by each recursion...
static const unsigned s_defaultSubdivideGrowRate = 50;
static PointCoordinateType s_maxSubdivideArea = 1;
static QMap<qint64, unsigned>
        s_alreadyCreatedVertices;  // map to store already created edges middle
                                   // points

static qint64 GenerateKey(unsigned edgeIndex1, unsigned edgeIndex2) {
    if (edgeIndex1 > edgeIndex2) std::swap(edgeIndex1, edgeIndex2);

    return (static_cast<qint64>(edgeIndex1) << 32) |
           static_cast<qint64>(edgeIndex2);
}

bool ccMesh::pushSubdivide(/*PointCoordinateType maxArea, */ unsigned indexA,
                           unsigned indexB,
                           unsigned indexC) {
    if (s_maxSubdivideArea /*maxArea*/ <= ZERO_TOLERANCE_F) {
        CVLog::Error("[ccMesh::pushSubdivide] Invalid input argument!");
        return false;
    }

    if (!getAssociatedCloud() ||
        !getAssociatedCloud()->isA(CV_TYPES::POINT_CLOUD)) {
        CVLog::Error(
                "[ccMesh::pushSubdivide] Vertices set must be a true point "
                "cloud!");
        return false;
    }
    ccPointCloud* vertices = static_cast<ccPointCloud*>(getAssociatedCloud());
    assert(vertices);
    const CCVector3* A = vertices->getPoint(indexA);
    const CCVector3* B = vertices->getPoint(indexB);
    const CCVector3* C = vertices->getPoint(indexC);

    // do we need to sudivide this triangle?
    PointCoordinateType area = ((*B - *A) * (*C - *A)).norm() / 2;
    if (area > s_maxSubdivideArea /*maxArea*/) {
        // we will add 3 new vertices, so we must be sure to have enough memory
        if (vertices->size() + 2 >= vertices->capacity()) {
            assert(s_defaultSubdivideGrowRate > 2);
            if (!vertices->reserve(vertices->size() +
                                   s_defaultSubdivideGrowRate)) {
                CVLog::Error("[ccMesh::pushSubdivide] Not enough memory!");
                return false;
            }
            // We have to update pointers as they may have been wrangled by the
            // 'reserve' call
            A = vertices->getPoint(indexA);
            B = vertices->getPoint(indexB);
            C = vertices->getPoint(indexC);
        }

        // add new vertices
        unsigned indexG1 = 0;
        {
            qint64 key = GenerateKey(indexA, indexB);
            QMap<qint64, unsigned>::const_iterator it =
                    s_alreadyCreatedVertices.constFind(key);
            if (it == s_alreadyCreatedVertices.constEnd()) {
                // generate new vertex
                indexG1 = vertices->size();
                CCVector3 G1 = (*A + *B) / 2;
                vertices->addPoint(G1);
                // interpolate other features?
                // if (vertices->hasNormals())
                //{
                //	//vertices->reserveTheNormsTable();
                //	CCVector3 N(0.0, 0.0, 1.0);
                //	interpolateNormals(indexA, indexB, indexC, G1, N);
                //	vertices->addNorm(N);
                // }
                if (vertices->hasColors()) {
                    ecvColor::Rgb C;
                    interpolateColors(cloudViewer::VerticesIndexes(
                                              indexA, indexB, indexC),
                                      G1, C);
                    vertices->addRGBColor(C);
                }
                // and add it to the map
                s_alreadyCreatedVertices.insert(key, indexG1);
            } else {
                indexG1 = it.value();
            }
        }
        unsigned indexG2 = 0;
        {
            qint64 key = GenerateKey(indexB, indexC);
            QMap<qint64, unsigned>::const_iterator it =
                    s_alreadyCreatedVertices.constFind(key);
            if (it == s_alreadyCreatedVertices.constEnd()) {
                // generate new vertex
                indexG2 = vertices->size();
                CCVector3 G2 = (*B + *C) / 2;
                vertices->addPoint(G2);
                // interpolate other features?
                // if (vertices->hasNormals())
                //{
                //	//vertices->reserveTheNormsTable();
                //	CCVector3 N(0.0, 0.0, 1.0);
                //	interpolateNormals(indexA, indexB, indexC, G2, N);
                //	vertices->addNorm(N);
                // }
                if (vertices->hasColors()) {
                    ecvColor::Rgb C;
                    interpolateColors(cloudViewer::VerticesIndexes(
                                              indexA, indexB, indexC),
                                      G2, C);
                    vertices->addRGBColor(C);
                }
                // and add it to the map
                s_alreadyCreatedVertices.insert(key, indexG2);
            } else {
                indexG2 = it.value();
            }
        }
        unsigned indexG3 = vertices->size();
        {
            qint64 key = GenerateKey(indexC, indexA);
            QMap<qint64, unsigned>::const_iterator it =
                    s_alreadyCreatedVertices.constFind(key);
            if (it == s_alreadyCreatedVertices.constEnd()) {
                // generate new vertex
                indexG3 = vertices->size();
                CCVector3 G3 = (*C + *A) / 2.0;
                vertices->addPoint(G3);
                // interpolate other features?
                // if (vertices->hasNormals())
                //{
                //	//vertices->reserveTheNormsTable();
                //	CCVector3 N(0.0, 0.0, 1.0);
                //	interpolateNormals(indexA, indexB, indexC, G3, N);
                //	vertices->addNorm(N);
                // }
                if (vertices->hasColors()) {
                    ecvColor::Rgb C;
                    interpolateColors(cloudViewer::VerticesIndexes(
                                              indexA, indexB, indexC),
                                      G3, C);
                    vertices->addRGBColor(C);
                }
                // and add it to the map
                s_alreadyCreatedVertices.insert(key, indexG3);
            } else {
                indexG3 = it.value();
            }
        }

        // add new triangles
        if (!pushSubdivide(/*maxArea, */ indexA, indexG1, indexG3))
            return false;
        if (!pushSubdivide(/*maxArea, */ indexB, indexG2, indexG1))
            return false;
        if (!pushSubdivide(/*maxArea, */ indexC, indexG3, indexG2))
            return false;
        if (!pushSubdivide(/*maxArea, */ indexG1, indexG2, indexG3))
            return false;
    } else {
        // we will add one triangle, so we must be sure to have enough memory
        if (size() == capacity()) {
            if (!reserve(size() + 3 * s_defaultSubdivideGrowRate)) {
                CVLog::Error("[ccMesh::pushSubdivide] Not enough memory!");
                return false;
            }
        }

        // we keep this triangle as is
        addTriangle(indexA, indexB, indexC);
    }

    return true;
}

ccMesh* ccMesh::subdivide(PointCoordinateType maxArea) const {
    if (cloudViewer::LessThanEpsilon(maxArea)) {
        CVLog::Error("[ccMesh::subdivide] Invalid input argument!");
        return nullptr;
    }
    s_maxSubdivideArea = maxArea;

    unsigned triCount = size();
    ccGenericPointCloud* vertices = getAssociatedCloud();
    unsigned vertCount = (vertices ? vertices->size() : 0);
    if (!vertices || vertCount * triCount == 0) {
        CVLog::Error("[ccMesh::subdivide] Invalid mesh: no face or no vertex!");
        return nullptr;
    }

    ccPointCloud* resultVertices =
            vertices->isA(CV_TYPES::POINT_CLOUD)
                    ? static_cast<ccPointCloud*>(vertices)->cloneThis()
                    : ccPointCloud::From(vertices, vertices);
    if (!resultVertices) {
        CVLog::Error("[ccMesh::subdivide] Not enough memory!");
        return nullptr;
    }

    ccMesh* resultMesh = new ccMesh(resultVertices);
    resultMesh->addChild(resultVertices);

    if (!resultMesh->reserve(triCount)) {
        CVLog::Error("[ccMesh::subdivide] Not enough memory!");
        delete resultMesh;
        return nullptr;
    }

    s_alreadyCreatedVertices.clear();

    try {
        for (unsigned i = 0; i < triCount; ++i) {
            const cloudViewer::VerticesIndexes& tri =
                    m_triVertIndexes->getValue(i);
            if (!resultMesh->pushSubdivide(/*maxArea,*/ tri.i1, tri.i2,
                                           tri.i3)) {
                CVLog::Error("[ccMesh::subdivide] Not enough memory!");
                delete resultMesh;
                return nullptr;
            }
        }
    } catch (...) {
        CVLog::Error("[ccMesh::subdivide] An error occurred!");
        delete resultMesh;
        return nullptr;
    }

    // we must also 'fix' the triangles that share (at least) an edge with a
    // subdivided triangle!
    try {
        unsigned newTriCount = resultMesh->size();
        for (unsigned i = 0; i < newTriCount; ++i) {
            cloudViewer::VerticesIndexes& tri =
                    resultMesh->m_triVertIndexes->getValue(
                            i);  // warning: array might change at each call to
                                 // reallocate!
            unsigned indexA = tri.i1;
            unsigned indexB = tri.i2;
            unsigned indexC = tri.i3;

            // test all edges
            int indexG1 = -1;
            {
                QMap<qint64, unsigned>::const_iterator it =
                        s_alreadyCreatedVertices.constFind(
                                GenerateKey(indexA, indexB));
                if (it != s_alreadyCreatedVertices.constEnd())
                    indexG1 = (int)it.value();
            }
            int indexG2 = -1;
            {
                QMap<qint64, unsigned>::const_iterator it =
                        s_alreadyCreatedVertices.constFind(
                                GenerateKey(indexB, indexC));
                if (it != s_alreadyCreatedVertices.constEnd())
                    indexG2 = (int)it.value();
            }
            int indexG3 = -1;
            {
                QMap<qint64, unsigned>::const_iterator it =
                        s_alreadyCreatedVertices.constFind(
                                GenerateKey(indexC, indexA));
                if (it != s_alreadyCreatedVertices.constEnd())
                    indexG3 = (int)it.value();
            }

            // at least one edge is 'wrong'
            unsigned brokenEdges = (indexG1 < 0 ? 0 : 1) +
                                   (indexG2 < 0 ? 0 : 1) +
                                   (indexG3 < 0 ? 0 : 1);

            if (brokenEdges == 1) {
                int indexG = indexG1;
                unsigned char i1 = 2;  // relative index facing the broken edge
                if (indexG2 >= 0) {
                    indexG = indexG2;
                    i1 = 0;
                } else if (indexG3 >= 0) {
                    indexG = indexG3;
                    i1 = 1;
                }
                assert(indexG >= 0);
                assert(i1 < 3);

                unsigned indexes[3] = {indexA, indexB, indexC};

                // replace current triangle by one half
                tri.i1 = indexes[i1];
                tri.i2 = indexG;
                tri.i3 = indexes[(i1 + 2) % 3];
                // and add the other half (we can use pushSubdivide as the area
                // should alredy be ok!)
                if (!resultMesh->pushSubdivide(/*maxArea,*/ indexes[i1],
                                               indexes[(i1 + 1) % 3], indexG)) {
                    CVLog::Error("[ccMesh::subdivide] Not enough memory!");
                    delete resultMesh;
                    return nullptr;
                }
            } else if (brokenEdges == 2) {
                if (indexG1 < 0)  // broken edges: BC and CA
                {
                    // replace current triangle by the 'pointy' part
                    tri.i1 = indexC;
                    tri.i2 = indexG3;
                    tri.i3 = indexG2;
                    // split the remaining 'trapezoid' in 2
                    if (!resultMesh->pushSubdivide(/*maxArea, */ indexA,
                                                   indexG2, indexG3) ||
                        !resultMesh->pushSubdivide(/*maxArea, */ indexA, indexB,
                                                   indexG2)) {
                        CVLog::Error("[ccMesh::subdivide] Not enough memory!");
                        delete resultMesh;
                        return nullptr;
                    }
                } else if (indexG2 < 0)  // broken edges: AB and CA
                {
                    // replace current triangle by the 'pointy' part
                    tri.i1 = indexA;
                    tri.i2 = indexG1;
                    tri.i3 = indexG3;
                    // split the remaining 'trapezoid' in 2
                    if (!resultMesh->pushSubdivide(/*maxArea, */ indexB,
                                                   indexG3, indexG1) ||
                        !resultMesh->pushSubdivide(/*maxArea, */ indexB, indexC,
                                                   indexG3)) {
                        CVLog::Error("[ccMesh::subdivide] Not enough memory!");
                        delete resultMesh;
                        return nullptr;
                    }
                } else /*if (indexG3 < 0)*/  // broken edges: AB and BC
                {
                    // replace current triangle by the 'pointy' part
                    tri.i1 = indexB;
                    tri.i2 = indexG2;
                    tri.i3 = indexG1;
                    // split the remaining 'trapezoid' in 2
                    if (!resultMesh->pushSubdivide(/*maxArea, */ indexC,
                                                   indexG1, indexG2) ||
                        !resultMesh->pushSubdivide(/*maxArea, */ indexC, indexA,
                                                   indexG1)) {
                        CVLog::Error("[ccMesh::subdivide] Not enough memory!");
                        delete resultMesh;
                        return nullptr;
                    }
                }
            } else if (brokenEdges ==
                       3)  // works just as a standard subdivision in fact!
            {
                // replace current triangle by one quarter
                tri.i1 = indexA;
                tri.i2 = indexG1;
                tri.i3 = indexG3;
                // and add the other 3 quarters (we can use pushSubdivide as the
                // area should alredy be ok!)
                if (!resultMesh->pushSubdivide(/*maxArea, */ indexB, indexG2,
                                               indexG1) ||
                    !resultMesh->pushSubdivide(/*maxArea, */ indexC, indexG3,
                                               indexG2) ||
                    !resultMesh->pushSubdivide(/*maxArea, */ indexG1, indexG2,
                                               indexG3)) {
                    CVLog::Error("[ccMesh::subdivide] Not enough memory!");
                    delete resultMesh;
                    return nullptr;
                }
            }
        }
    } catch (...) {
        CVLog::Error("[ccMesh::subdivide] An error occurred!");
        delete resultMesh;
        return nullptr;
    }

    s_alreadyCreatedVertices.clear();

    resultMesh->shrinkToFit();
    resultVertices->shrinkToFit();

    // we import from the original mesh... what we can
    if (hasNormals()) {
        if (hasNormals())  // normals interpolation doesn't work well...
            resultMesh->computeNormals(!hasTriNormals());
        resultMesh->showNormals(normalsShown());
    }
    if (hasColors()) {
        resultMesh->showColors(colorsShown());
    }
    resultMesh->setVisible(isVisible());

    return resultMesh;
}

bool ccMesh::convertMaterialsToVertexColors() {
    if (!hasMaterials()) {
        CVLog::Warning(
                "[ccMesh::convertMaterialsToVertexColors] Mesh has no "
                "material!");
        return false;
    }

    if (!m_associatedCloud->isA(CV_TYPES::POINT_CLOUD)) {
        CVLog::Warning(
                "[ccMesh::convertMaterialsToVertexColors] Need a true point "
                "cloud as vertices!");
        return false;
    }

    ccPointCloud* cloud = static_cast<ccPointCloud*>(m_associatedCloud);
    if (!cloud->resizeTheRGBTable(true)) {
        CVLog::Warning(
                "[ccMesh::convertMaterialsToVertexColors] Failed to resize "
                "vertices color table! (not enough memory?)");
        return false;
    }

    // now scan all faces and get the vertex color each time
    unsigned faceCount = size();

    placeIteratorAtBeginning();
    for (unsigned i = 0; i < faceCount; ++i) {
        const cloudViewer::VerticesIndexes* tsi = getNextTriangleVertIndexes();
        for (unsigned char j = 0; j < 3; ++j) {
            ecvColor::Rgb C;
            if (getVertexColorFromMaterial(i, j, C, true)) {
                // FIXME: could we be smarter? (we process each point several
                // times! And we assume the color is always the same...)
                cloud->setPointColor(tsi->i[j], C);
            }
        }
    }

    return true;
}
