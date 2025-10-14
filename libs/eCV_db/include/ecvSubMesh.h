// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SUB_MESH_HEADER
#define ECV_SUB_MESH_HEADER

// Local
#include "ecvBBox.h"
#include "ecvGenericMesh.h"

class ccMesh;

//! A sub-mesh
/** Equivalent to a cloudViewer::ReferenceCloud for a mesh
 **/
class ECV_DB_LIB_API ccSubMesh : public ccGenericMesh {
public:
    //! Default constructor
    explicit ccSubMesh(ccMesh* parentMesh);
    //! Destructor
    ~ccSubMesh() override = default;

    //! Returns class ID
    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::SUB_MESH; }

    // inherited methods (ccHObject)
    ccBBox getOwnBB(bool withGLFeatures = false) override;
    bool isSerializable() const override { return true; }

    // inherited methods (ccGenericMesh)
    ccGenericPointCloud* getAssociatedCloud() const override;
    void refreshBB() override;
    bool interpolateNormalsBC(unsigned triIndex,
                              const CCVector3d& w,
                              CCVector3& N) override;
    bool interpolateColors(unsigned triIndex,
                           const CCVector3& P,
                           ecvColor::Rgb& rgb) override;
    bool getColorFromMaterial(unsigned triIndex,
                              const CCVector3& P,
                              ecvColor::Rgb& rgb,
                              bool interpolateColorIfNoTexture) override;
    bool getVertexColorFromMaterial(unsigned triIndex,
                                    unsigned char vertIndex,
                                    ecvColor::Rgb& rgb,
                                    bool returnColorIfNoTexture) override;
    bool hasMaterials() const override;
    const ccMaterialSet* getMaterialSet() const override;
    int getTriangleMtlIndex(unsigned triangleIndex) const override;
    bool hasTextures() const override;
    TextureCoordsContainer* getTexCoordinatesTable() const override;
    void getTexCoordinates(unsigned index, TexCoords2D*& tx) const override;
    void getTriangleTexCoordinates(unsigned triIndex,
                                   TexCoords2D*& tx1,
                                   TexCoords2D*& tx2,
                                   TexCoords2D*& tx3) const override;
    bool hasPerTriangleTexCoordIndexes() const override;
    void getTriangleTexCoordinatesIndexes(unsigned triangleIndex,
                                          int& i1,
                                          int& i2,
                                          int& i3) const override;
    bool hasTriNormals() const override;
    void getTriangleNormalIndexes(unsigned triangleIndex,
                                  int& i1,
                                  int& i2,
                                  int& i3) const override;
    bool getTriangleNormals(unsigned triangleIndex,
                            CCVector3& Na,
                            CCVector3& Nb,
                            CCVector3& Nc) const override;
    NormsIndexesTableType* getTriNormsTable() const override;
    unsigned capacity() const override;

    // inherited methods (ccDrawableObject)
    bool hasColors() const override;
    bool hasNormals() const override;
    bool hasScalarFields() const override;
    bool hasDisplayedScalarField() const override;
    bool normalsShown() const override;

    // inherited methods (GenericIndexedMesh)
    inline unsigned size() const override {
        return static_cast<unsigned>(m_triIndexes.size());
    }
    void forEach(genericTriangleAction action) override;
    inline void placeIteratorAtBeginning() override { m_globalIterator = 0; }
    cloudViewer::GenericTriangle* _getNextTriangle()
            override;  // temporary object
    cloudViewer::GenericTriangle* _getTriangle(
            unsigned index) override;  // temporary object
    cloudViewer::VerticesIndexes* getNextTriangleVertIndexes() override;
    cloudViewer::VerticesIndexes* getTriangleVertIndexes(
            unsigned triangleIndex) override;
    virtual void getTriangleVertices(unsigned triangleIndex,
                                     CCVector3& A,
                                     CCVector3& B,
                                     CCVector3& C) const override;
    virtual void getTriangleVertices(unsigned triangleIndex,
                                     double A[3],
                                     double B[3],
                                     double C[3]) const override;

    void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) override;
    bool interpolateNormals(unsigned triIndex,
                            const CCVector3& P,
                            CCVector3& N) override;

    //! Returns global index (i.e. relative to the associated mesh) of a given
    //! element
    /** \param localIndex local index (i.e. relative to the internal index
     *container)
     **/
    inline unsigned getTriGlobalIndex(unsigned localIndex) const {
        return m_triIndexes[localIndex];
    }

    //! Returns the global index of the triangle pointed by the current element
    inline unsigned getCurrentTriGlobalIndex() const {
        assert(m_globalIterator < size());
        return m_triIndexes[m_globalIterator];
    }

    //! Forwards the local element iterator
    inline void forwardIterator() { ++m_globalIterator; }

    //! Clears the mesh
    void clear(bool releaseMemory);

    //! Triangle global index insertion mechanism
    /** \param globalIndex a triangle global index
            \return false if not enough memory
    **/
    bool addTriangleIndex(unsigned globalIndex);

    //! Triangle global index insertion mechanism (range)
    /** \param firstIndex first triangle global index of range
            \param lastIndex last triangle global index of range (excluded)
            \return false if not enough memory
    **/
    bool addTriangleIndex(unsigned firstIndex, unsigned lastIndex);

    //! Sets global index for a given element
    /** \param localIndex local index
            \param globalIndex global index
    **/
    void setTriangleIndex(unsigned localIndex, unsigned globalIndex);

    //! Reserves some memory for hosting the triangle references
    /** \param n the number of triangles (references)
     **/
    bool reserve(size_t n);

    //! Presets the size of the vector used to store triangle references
    /** \param n the number of triangles (references)
     **/
    bool resize(size_t n);

    //! Returns the associated mesh
    inline ccMesh* getAssociatedMesh() { return m_associatedMesh; }

    //! Returns the associated mesh (const version)
    inline const ccMesh* getAssociatedMesh() const { return m_associatedMesh; }

    //! Sets the associated mesh
    /** \param mesh parent mesh
            \param unlinkPreviousOne whether to remove any dependency with the
    previous parent mesh (if any)
    **/
    void setAssociatedMesh(ccMesh* mesh, bool unlinkPreviousOne = true);

    //! Indexes map for createNewSubMeshFromSelection
    using IndexMap = std::vector<unsigned int>;

    //! Creates a new sub mesh with the visible vertices only
    /** \param removeSelectedTriangles specifies if the faces composed only of
    'selected' vertices should be removed or not \param selectedTriangleIndexes
    a map of the triangles indexes that will be used in the new sub msh (or -1
    if they are discarded) \param newRemainingTriangleIndexes if an index map is
    provided, it will be used to 'translate' global indexes of triangles
    remaining in the source sub-mesh \return the new sub-mesh if successful
    **/
    ccSubMesh* createNewSubMeshFromSelection(
            bool removeSelectedTriangles,
            const std::vector<int>& selectedTriangleIndexes,
            IndexMap* newRemainingTriangleIndexes = nullptr);

    //! Creates a new sub mesh with the selected vertices only
    /** This method is called after a graphical segmentation
            or by ccMesh::createNewMeshFromSelection.
            It creates a new sub-mesh structure with the vertices that are
            tagged as "visible" (see ccGenericPointCloud::visibilityArray).
            This method will also update this sub-mesh if removeSelectedFaces is
    true. In this case, all "selected" triangles will be removed from this
    sub-mesh's instance.

            \param removeSelectedFaces specifies if the faces composed only of
    'selected' vertices should be removed or not \param indexMap if an index map
    is provided, it will be used to 'translate' global indexes
    **/
    ccSubMesh* createNewSubMeshFromSelection(bool removeSelectedFaces,
                                             IndexMap* indexMap = nullptr);

protected:
    // inherited from ccHObject
    bool toFile_MeOnly(QFile& out) const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
    void onUpdateOf(ccHObject* obj) override;

    //! Associated mesh
    ccMesh* m_associatedMesh;

    //! Container of 3D triangles indexes
    using ReferencesContainer = std::vector<unsigned int>;

    //! Indexes of (some of) the associated mesh triangles
    ReferencesContainer m_triIndexes;

    //! Iterator on the triangles references container
    unsigned m_globalIterator;

    //! Bounding-box
    ccBBox m_bBox;
};

#endif  // ECV_SUB_MESH_HEADER
