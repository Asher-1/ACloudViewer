// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <Helper.h>
#include <PointProjectionTools.h>
#include <SimpleTriangle.h>

#include <Eigen/Core>

// Local
#include <unordered_map>
#include <unordered_set>

#include "Image.h"
#include "ecvGenericMesh.h"

namespace cloudViewer {
namespace geometry {
class TetraMesh;
}
}  // namespace cloudViewer

class ccPolyline;
class ecvOrientedBBox;
class ecvProgressDialog;

/**
 * @class ccMesh
 * @brief Triangular mesh representation
 *
 * Implements a triangular mesh structure with support for:
 * - Vertex positions (from associated point cloud)
 * - Per-vertex normals
 * - Texture coordinates
 * - Materials
 * - Triangle-based topology
 *
 * The mesh references an external point cloud for vertex positions,
 * allowing efficient memory management and data sharing.
 */
class CV_DB_LIB_API ccMesh : public ccGenericMesh {
public:
    /**
     * @brief Default constructor
     * @param vertices Associated vertices cloud (optional)
     */
    explicit ccMesh(ccGenericPointCloud* vertices = nullptr);

    /**
     * @brief Copy constructor
     * @param mesh Source mesh to copy
     */
    ccMesh(const ccMesh& mesh);

    /**
     * @brief Parameterized constructor from vertex and triangle lists
     * @param vertices List of vertex positions
     * @param triangles List of triangles (each triangle = 3 vertex indices)
     */
    explicit ccMesh(const std::vector<Eigen::Vector3d>& vertices,
                    const std::vector<Eigen::Vector3i>& triangles);

    /**
     * @brief Constructor from GenericIndexedMesh
     *
     * Creates a ccMesh from a generic indexed mesh structure.
     * @param giMesh Generic indexed mesh
     * @param giVertices Vertex cloud for the mesh
     * @note The GenericIndexedMesh should reference a known ccGenericPointCloud
     */
    explicit ccMesh(cloudViewer::GenericIndexedMesh* giMesh,
                    ccGenericPointCloud* giVertices);

    /**
     * @brief Destructor
     */
    ~ccMesh() override;

    /**
     * @brief Get class ID
     * @return Class identifier (CV_TYPES::MESH)
     */
    CV_CLASS_ENUM getClassID() const override { return CV_TYPES::MESH; }

    /**
     * @brief Set the associated vertices cloud
     * @param cloud Vertices cloud to associate
     * @warning Changing the associated cloud may invalidate existing data
     */
    void setAssociatedCloud(ccGenericPointCloud* cloud);

    /**
     * @brief Create internal vertices cloud
     *
     * Creates and initializes an internal point cloud for storing vertices.
     * @return true if successful
     */
    bool CreateInternalCloud();

    /**
     * @brief Clone the mesh
     *
     * Creates a deep copy of the mesh with all features except the octree.
     * @param vertices Vertices to use (auto-cloned if nullptr)
     * @param clonedMaterials Cloned materials (for internal use)
     * @param clonedNormsTable Cloned normals table (for internal use)
     * @param cloneTexCoords Cloned texture coordinates (for internal use)
     * @return Cloned mesh
     */
    ccMesh* cloneMesh(ccGenericPointCloud* vertices = nullptr,
                      ccMaterialSet* clonedMaterials = nullptr,
                      NormsIndexesTableType* clonedNormsTable = nullptr,
                      TextureCoordsContainer* cloneTexCoords = nullptr);

    /**
     * @brief Create partial mesh from triangle selection
     *
     * Similar to ccPointCloud::partialClone but for meshes.
     * @param triangleIndices Indices of triangles to include
     * @param warnings Optional pointer to store warning flags
     * @return Partial clone (nullptr on error)
     */
    ccMesh* partialClone(const std::vector<unsigned>& triangleIndices,
                         int* warnings = nullptr) const;

    /**
     * @brief Get ordered triangle vertices
     *
     * Returns triangle vertex indices in ascending order.
     * @param vidx0 First vertex index
     * @param vidx1 Second vertex index
     * @param vidx2 Third vertex index
     * @return Ordered vertex indices
     */
    static inline cloudViewer::VerticesIndexes GetOrderedTriangle(int vidx0,
                                                                  int vidx1,
                                                                  int vidx2) {
        if (vidx0 > vidx2) {
            std::swap(vidx0, vidx2);
        }
        if (vidx0 > vidx1) {
            std::swap(vidx0, vidx1);
        }
        if (vidx1 > vidx2) {
            std::swap(vidx1, vidx2);
        }
        return cloudViewer::VerticesIndexes(static_cast<unsigned int>(vidx0),
                                            static_cast<unsigned int>(vidx1),
                                            static_cast<unsigned int>(vidx2));
    }

    /**
     * @brief Create Delaunay 2.5D triangulation from point cloud
     *
     * Performs 2D or 2.5D Delaunay triangulation on a point cloud.
     * @param cloud Input point cloud
     * @param type Triangulation type
     * @param updateNormals Whether to compute normals (default: false)
     * @param maxEdgeLength Maximum edge length (0 for unlimited)
     * @param dim Projection dimension (default: 2)
     * @return Triangulated mesh (nullptr on error)
     * @see cloudViewer::PointProjectionTools::computeTriangulation
     */
    static ccMesh* Triangulate(ccGenericPointCloud* cloud,
                               cloudViewer::TRIANGULATION_TYPES type,
                               bool updateNormals = false,
                               PointCoordinateType maxEdgeLength = 0,
                               unsigned char dim = 2);

    /**
     * @brief Create mesh from two polylines
     *
     * Triangulates the region between two polylines.
     * @param p1 First polyline
     * @param p2 Second polyline
     * @param projectionDir Optional projection direction
     * @return Triangulated mesh (nullptr on error)
     */
    static ccMesh* TriangulateTwoPolylines(ccPolyline* p1,
                                           ccPolyline* p2,
                                           CCVector3* projectionDir = nullptr);

    /**
     * @brief Merge another mesh into this one
     * @param mesh Mesh to merge
     * @param createSubMesh Whether to create a submesh entity
     * @return true if successful
     */
    bool merge(const ccMesh* mesh, bool createSubMesh);

    ccMesh& operator=(const ccMesh& mesh);
    ccMesh& operator+=(const ccMesh& mesh);
    ccMesh operator+(const ccMesh& mesh) const;

    void clear();

    // inherited methods (ccHObject)
    unsigned getUniqueIDForDisplay() const override;
    ccBBox getOwnBB(bool withGLFeatures = false) override;
    bool isSerializable() const override { return true; }
    const ccGLMatrix& getGLTransformationHistory() const override;

    // inherited methods (ccGenericMesh)
    inline ccGenericPointCloud* getAssociatedCloud() const override {
        return m_associatedCloud;
    }
    void refreshBB() override;
    bool interpolateNormalsBC(unsigned triIndex,
                              const CCVector3d& w,
                              CCVector3& N) override;
    bool interpolateColors(unsigned triIndex,
                           const CCVector3& P,
                           ecvColor::Rgb& C) override;
    void computeInterpolationWeights(unsigned triIndex,
                                     const CCVector3& P,
                                     CCVector3d& weights) const override;
    bool getColorFromMaterial(unsigned triIndex,
                              const CCVector3& P,
                              ecvColor::Rgb& C,
                              bool interpolateColorIfNoTexture) override;
    bool getVertexColorFromMaterial(unsigned triIndex,
                                    unsigned char vertIndex,
                                    ecvColor::Rgb& C,
                                    bool returnColorIfNoTexture) override;
    unsigned capacity() const override;

    // inherited methods (GenericIndexedMesh)
    void forEach(genericTriangleAction action) override;
    void placeIteratorAtBeginning() override;
    cloudViewer::GenericTriangle* _getNextTriangle() override;  // temporary
    cloudViewer::GenericTriangle* _getTriangle(
            unsigned triangleIndex) override;  // temporary
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

    unsigned int getVerticeSize() const;

    Eigen::Vector3d getVertice(size_t index) const;
    void setVertice(size_t index, const Eigen::Vector3d& vertice);
    void addVertice(const Eigen::Vector3d& vertice);
    std::vector<Eigen::Vector3d> getEigenVertices() const;
    void addEigenVertices(const std::vector<Eigen::Vector3d>& vertices);
    void setEigenVertices(const std::vector<Eigen::Vector3d>& vertices);

    Eigen::Vector3d getVertexNormal(size_t index) const;
    void setVertexNormal(size_t index, const Eigen::Vector3d& normal);
    void addVertexNormal(const Eigen::Vector3d& normal);
    std::vector<Eigen::Vector3d> getVertexNormals() const;
    void addVertexNormals(const std::vector<Eigen::Vector3d>& normals);
    void setVertexNormals(const std::vector<Eigen::Vector3d>& normals);

    Eigen::Vector3d getVertexColor(size_t index) const;
    void setVertexColor(size_t index, const Eigen::Vector3d& color);
    void addVertexColor(const Eigen::Vector3d& color);
    std::vector<Eigen::Vector3d> getVertexColors() const;
    ColorsTableType* getVertexColorsPtr();
    void addVertexColors(const std::vector<Eigen::Vector3d>& colors);
    void setVertexColors(const std::vector<Eigen::Vector3d>& colors);

    inline bool HasVertices() const { return getVerticeSize() != 0; }

    std::vector<CCVector3>& getVerticesPtr();
    const std::vector<CCVector3>& getVertices() const;

    virtual unsigned size() const override;
    void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) override;
    bool normalsAvailable() const override { return hasNormals(); }
    bool interpolateNormals(unsigned triIndex,
                            const CCVector3& P,
                            CCVector3& N) override;

    // const version of getTriangleVertIndexes
    void getTriangleVertIndexes(size_t triangleIndex,
                                Eigen::Vector3i& vertIndx) const;
    const virtual cloudViewer::VerticesIndexes* getTriangleVertIndexes(
            unsigned triangleIndex) const;

    // inherited methods (ccDrawableObject)
    bool hasColors() const override;
    bool hasNormals() const override;
    bool HasVertexNormals() const;
    bool hasScalarFields() const override;
    bool hasDisplayedScalarField() const override;
    bool normalsShown() const override;
    void toggleMaterials() override { showMaterials(!materialsShown()); }

    //! Inverts normals (if any)
    /** Either the per-triangle normals, or the per-vertex ones
     **/
    void invertNormals();

    //! Shifts all triangles indexes
    /** \param shift index shift (positive)
     **/
    void shiftTriangleIndexes(unsigned shift);

    //! Flips the triangle
    /** Swaps the second and third vertices indexes
     **/
    void flipTriangles();

    //! Adds a triangle to the mesh
    /** \warning Bounding-box validity is broken after a call to this method.
            However, for the sake of performance, no call to
    notifyGeometryUpdate is made automatically. Make sure to do so when all
    modifications are done! \param i1 first vertex index (relatively to the
    vertex cloud) \param i2 second vertex index (relatively to the vertex cloud)
            \param i3 third vertex index (relatively to the vertex cloud)
    **/
    void addTriangle(unsigned i1, unsigned i2, unsigned i3);
    void addTriangle(const cloudViewer::VerticesIndexes& triangle);
    inline void addTriangle(const Eigen::Vector3i& index) {
        addTriangle(static_cast<unsigned>(index[0]),
                    static_cast<unsigned>(index[1]),
                    static_cast<unsigned>(index[2]));
    }
    inline void addTriangles(const std::vector<Eigen::Vector3i>& triangles) {
        for (auto& tri : triangles) {
            addTriangle(tri);
        }
    }

    void setTriangle(size_t index, const Eigen::Vector3i& triangle);
    void setTriangles(const std::vector<Eigen::Vector3i>& triangles);
    Eigen::Vector3i getTriangle(size_t index) const;
    std::vector<Eigen::Vector3i> getTriangles() const;

    //! Container of per-triangle vertices indexes (3)
    using triangleIndexesContainer =
            ccArray<cloudViewer::VerticesIndexes, 3, unsigned>;
    inline triangleIndexesContainer* getTrianglesPtr() const {
        return m_triVertIndexes;
    }

    //! Reserves the memory to store the vertex indexes (3 per triangle)
    /** \param n the number of triangles to reserve
            \return true if the method succeeds, false otherwise
    **/
    bool reserve(std::size_t n);
    bool reserveAssociatedCloud(std::size_t n,
                                bool init_color = false,
                                bool init_normal = false);

    //! Resizes the array of vertex indexes (3 per triangle)
    /** If the new number of elements is smaller than the actual size,
            the overflooding elements will be deleted.
            \param n the new number of triangles
            \return true if the method succeeds, false otherwise
    **/
    bool resize(size_t n);
    bool resizeAssociatedCloud(std::size_t n);

    //! Removes unused capacity
    inline void shrinkToFit() {
        if (size() < capacity()) resize(size());
    }
    void shrinkVertexToFit();

    /*********************************************************/
    /**************    PER-TRIANGLE NORMALS    ***************/
    /*********************************************************/

    // inherited from ccGenericMesh
    bool hasTriNormals() const override;
    // for compatibility
    inline bool HasTriangleNormals() const { return hasTriNormals(); }
    void getTriangleNormalIndexes(unsigned triangleIndex,
                                  int& i1,
                                  int& i2,
                                  int& i3) const override;
    bool getTriangleNormals(unsigned triangleIndex,
                            CCVector3& Na,
                            CCVector3& Nb,
                            CCVector3& Nc) const override;
    bool getTriangleNormals(unsigned triangleIndex,
                            double Na[3],
                            double Nb[3],
                            double Nc[3]) const override;
    bool getTriangleNormals(unsigned triangleIndex,
                            Eigen::Vector3d& Na,
                            Eigen::Vector3d& Nb,
                            Eigen::Vector3d& Nc) const override;
    std::vector<Eigen::Vector3d> getTriangleNormals() const;
    std::vector<CCVector3*> getTriangleNormalsPtr() const;
    Eigen::Vector3d getTriangleNorm(size_t index) const;
    bool setTriangleNorm(size_t index, const Eigen::Vector3d& triangle_normal);
    bool setTriangleNormalIndexes(size_t triangleIndex,
                                  CompressedNormType value);
    CompressedNormType getTriangleNormalIndexes(size_t triangleIndex);
    bool addTriangleNorm(const CCVector3& N);
    bool addTriangleNorm(const Eigen::Vector3d& N);
    std::vector<Eigen::Vector3d> getTriangleNorms() const;
    bool setTriangleNorms(const std::vector<Eigen::Vector3d>& triangle_normals);
    bool addTriangleNorms(const std::vector<Eigen::Vector3d>& triangle_normals);

    NormsIndexesTableType* getTriNormsTable() const override {
        return m_triNormals;
    }

    //! Sets per-triangle normals array (may be shared)
    void setTriNormsTable(NormsIndexesTableType* triNormsTable,
                          bool autoReleaseOldTable = true);

    //! Removes per-triangle normals
    void clearTriNormals() { setTriNormsTable(nullptr); }

    //! Returns whether per triangle normals are enabled
    /** To enable per triangle normals, you should:
            - first, reserve memory for triangles (this is always the first
    thing to do)
            - associate this mesh to a triangle normals array (see
    ccMesh::setTriNormsTable)
            - reserve memory to store per-triangle normal indexes with
    ccMesh::reservePerTriangleNormalIndexes
            - add for each triangle a triplet of indexes (referring to stored
    normals)
    **/
    bool arePerTriangleNormalsEnabled() const;

    //! Reserves memory to store per-triangle triplets of normal indexes
    /** Before adding per-triangle normal indexes triplets to
            the mesh (with ccMesh::addTriangleNormalsIndexes()) be
            sure to reserve the  necessary amount of memory with
            this method. This method reserves memory for as many
            normals indexes triplets as the number of triangles
            in the mesh (effictively stored or reserved - a call to
            ccMesh::reserve prior to this one is mandatory).
            \return true if ok, false if there's not enough memory
    **/
    bool reservePerTriangleNormalIndexes();

    //! Adds a triplet of normal indexes for next triangle
    /** Make sure per-triangle normal indexes array is allocated
            (see reservePerTriangleNormalIndexes)
            \param i1 first vertex normal index
            \param i2 second vertex normal index
            \param i3 third vertex normal index
    **/
    void addTriangleNormalIndexes(int i1, int i2, int i3);

    //! Sets a triplet of normal indexes for a given triangle
    /** \param triangleIndex triangle index
            \param i1 first vertex normal index
            \param i2 second vertex normal index
            \param i3 third vertex normal index
    **/
    void setTriangleNormalIndexes(unsigned triangleIndex,
                                  int i1,
                                  int i2,
                                  int i3);

    //! Removes any per-triangle triplets of normal indexes
    void removePerTriangleNormalIndexes();

    //! Invert per-triangle normals
    void invertPerTriangleNormals();

    /********************************************************/
    /************    PER-TRIANGLE MATERIAL    ***************/
    /********************************************************/

    // inherited from ccGenericMesh
    bool hasMaterials() const override;
    const ccMaterialSet* getMaterialSet() const override { return m_materials; }
    int getTriangleMtlIndex(unsigned triangleIndex) const override;

    //! Converts materials to vertex colors
    /** Warning: this method will overwrite colors (if any)
     **/
    bool convertMaterialsToVertexColors();

    //! Returns whether this mesh as per-triangle material index
    bool hasPerTriangleMtlIndexes() const {
        return m_triMtlIndexes && m_triMtlIndexes->isAllocated();
    }

    //! Reserves memory to store per-triangle material index
    /** Before adding per-triangle material index to
            the mesh (with ccMesh::addTriangleMtlIndex()) be sure
            to reserve the  necessary amount of memory with this
            method. This method reserves memory for as many
            material descriptors as the number of triangles in
            the mesh (effictively stored or reserved - a call to
            ccMesh::reserve prior to this one is mandatory).
            \return true if ok, false if there's not enough memory
    **/
    bool reservePerTriangleMtlIndexes();

    //! Removes any per-triangle material indexes
    void removePerTriangleMtlIndexes();

    //! Adds triangle material index for next triangle
    /** Cf. ccMesh::reservePerTriangleMtlIndexes.
            \param mtlIndex triangle material index
    **/
    void addTriangleMtlIndex(int mtlIndex);

    //! Container of per-triangle material descriptors
    using triangleMaterialIndexesSet = ccArray<int, 1, int>;

    //! Sets per-triangle material indexes array
    void setTriangleMtlIndexesTable(triangleMaterialIndexesSet* matIndexesTable,
                                    bool autoReleaseOldTable = true);

    //! Returns the per-triangle material indexes array
    inline const triangleMaterialIndexesSet* getTriangleMtlIndexesTable()
            const {
        return m_triMtlIndexes;
    }

    //! Sets triangle material indexes
    /** Cf. ccMesh::reservePerTriangleMtlIndexes.
            \param triangleIndex triangle index
            \param mtlIndex triangle material index
    **/
    void setTriangleMtlIndex(unsigned triangleIndex, int mtlIndex);

    //! Sets associated material set (may be shared)
    void setMaterialSet(ccMaterialSet* materialSet,
                        bool autoReleaseOldMaterialSet = true);

    /******************************************************************/
    /************    PER-TRIANGLE TEXTURE COORDINATE    ***************/
    /******************************************************************/

    // inherited from ccGenericMesh
    bool hasTextures() const override;
    TextureCoordsContainer* getTexCoordinatesTable() const override {
        return m_texCoords;
    }
    void getTriangleTexCoordinates(unsigned triIndex,
                                   TexCoords2D*& tx1,
                                   TexCoords2D*& tx2,
                                   TexCoords2D*& tx3) const override;
    void getTexCoordinates(unsigned index, TexCoords2D*& tx) const override;
    bool hasPerTriangleTexCoordIndexes() const override {
        return m_texCoordIndexes && m_texCoordIndexes->isAllocated();
    }
    void getTriangleTexCoordinatesIndexes(unsigned triangleIndex,
                                          int& i1,
                                          int& i2,
                                          int& i3) const override;

    //! Sets per-triangle texture coordinates array (may be shared)
    void setTexCoordinatesTable(TextureCoordsContainer* texCoordsTable,
                                bool autoReleaseOldTable = true);

    //! Reserves memory to store per-triangle triplets of tex coords indexes
    /** Before adding per-triangle tex coords indexes triplets to
            the mesh (with ccMesh::addTriangleTexCoordIndexes()) be
            sure to reserve the  necessary amount of memory with
            this method. This method reserves memory for as many
            tex coords indexes triplets as the number of triangles
            in the mesh (effictively stored or reserved - a call to
            ccMesh::reserve prior to this one is mandatory).
            \return true if ok, false if there's not enough memory
    **/
    bool reservePerTriangleTexCoordIndexes();

    //! Remove per-triangle tex coords indexes
    void removePerTriangleTexCoordIndexes();

    //! Adds a triplet of tex coords indexes for next triangle
    /** Make sure per-triangle tex coords indexes array is allocated
            (see reservePerTriangleTexCoordIndexes)
            \param i1 first vertex tex coords index
            \param i2 second vertex tex coords index
            \param i3 third vertex tex coords index
    **/
    void addTriangleTexCoordIndexes(int i1, int i2, int i3);

    //! Sets a triplet of tex coords indexes for a given triangle
    /** \param triangleIndex triangle index
            \param i1 first vertex tex coords index
            \param i2 second vertex tex coords index
            \param i3 third vertex tex coords index
    **/
    void setTriangleTexCoordIndexes(unsigned triangleIndex,
                                    int i1,
                                    int i2,
                                    int i3);

    //! Computes normals
    /** \param perVertex whether normals should be computed per-vertex or
    per-triangle \return success
    **/
    bool computeNormals(bool perVertex);

    //! Computes per-vertex normals
    bool computePerVertexNormals();

    //! Computes per-triangle normals
    bool computePerTriangleNormals();

    //! Laplacian smoothing
    /** \param nbIteration smoothing iterations
            \param factor smoothing 'force'
            \param progressCb progress dialog callback
    **/
    bool laplacianSmooth(
            unsigned nbIteration = 100,
            PointCoordinateType factor = static_cast<PointCoordinateType>(0.01),
            ecvProgressDialog* progressCb = nullptr);

    //! Mesh scalar field processes
    enum MESH_SCALAR_FIELD_PROCESS {
        SMOOTH_MESH_SF,  /**< Smooth **/
        ENHANCE_MESH_SF, /**< Enhance **/
    };

    //! Applies process to the mesh scalar field (the one associated to its
    //! vertices in fact)
    /** A very simple smoothing/enhancement algorithm based on
            each vertex direct neighbours. Prior to calling this method,
            one should check first that the vertices are associated to a
            scalar field.
            Warning: the processed scalar field must be enabled for both
            INPUT & OUTPUT! (see ccGenericCloud::setCurrentScalarField)
            \param process either 'smooth' or 'enhance'
    **/
    bool processScalarField(MESH_SCALAR_FIELD_PROCESS process);

    //! Subdivides mesh (so as to ensure that all triangles are falls below
    //! 'maxArea')
    /** \return subdivided mesh (if successful)
     **/
    ccMesh* subdivide(PointCoordinateType maxArea) const;

    //! Creates a new mesh with the selected vertices only
    /** This method is called after a graphical segmentation. It creates
            a new mesh structure with the vertices that are tagged as "visible"
            (see ccGenericPointCloud::visibilityArray).
            This method will also update this mesh if removeSelectedFaces is
    true. In this case, all "selected" triangles will be removed from this
    mesh's instance.

            \param	removeSelectedTriangles			specifies if the
    faces composed only of 'selected' vertices should be removed or not. If
    true, the visibility array will be automatically unallocated on completion
    \param newIndexesOfRemainingTriangles	the new indexes of the remaining
    triangles (if removeSelectedTriangles is true - optional). Must be initially
    empty or have the same size as the original mesh. \param
    withChildEntities whether child entities should be transferred as well (see
    ccHObjectCaster::CloneChildren) \return	the new mesh (if successful) or
    itself if all vertices were visible/selected
    **/
    ccMesh* createNewMeshFromSelection(
            bool removeSelectedTriangles,
            std::vector<int>* newIndexesOfRemainingTriangles = nullptr,
            bool withChildEntities = false);

    //! Swaps two triangles
    /** Automatically updates internal structures (i.e. lookup tables for
            material, normals, etc.).
    **/
    void swapTriangles(unsigned index1, unsigned index2);

    void removeTriangles(size_t index);

    //! Transforms the mesh per-triangle normals
    void transformTriNormals(const ccGLMatrix& trans);

    //! Default octree level for the 'mergeDuplicatedVertices' algorithm
    static const unsigned char DefaultMergeDulicateVerticesLevel = 10;

    //! Merges duplicated vertices
    bool mergeDuplicatedVertices(
            unsigned char octreeLevel = DefaultMergeDulicateVerticesLevel,
            QWidget* parentWidget = nullptr);

public:  // some cloudViewer interface
    /// The set adjacency_list[i] contains the indices of adjacent vertices of
    /// vertex i.
    std::vector<std::unordered_set<int>> adjacency_list_;

    /// List of uv coordinates per triangle.
    std::vector<Eigen::Vector2d> triangle_uvs_;

    struct Material {
        struct MaterialParameter {
            float f4[4] = {0};

            MaterialParameter() {
                f4[0] = 0;
                f4[1] = 0;
                f4[2] = 0;
                f4[3] = 0;
            }

            MaterialParameter(const float v1,
                              const float v2,
                              const float v3,
                              const float v4) {
                f4[0] = v1;
                f4[1] = v2;
                f4[2] = v3;
                f4[3] = v4;
            }

            MaterialParameter(const float v1, const float v2, const float v3) {
                f4[0] = v1;
                f4[1] = v2;
                f4[2] = v3;
                f4[3] = 1;
            }

            MaterialParameter(const float v1, const float v2) {
                f4[0] = v1;
                f4[1] = v2;
                f4[2] = 0;
                f4[3] = 0;
            }

            explicit MaterialParameter(const float v1) {
                f4[0] = v1;
                f4[1] = 0;
                f4[2] = 0;
                f4[3] = 0;
            }

            static MaterialParameter CreateRGB(const float r,
                                               const float g,
                                               const float b) {
                return {r, g, b, 1.f};
            }

            float r() const { return f4[0]; }
            float g() const { return f4[1]; }
            float b() const { return f4[2]; }
            float a() const { return f4[3]; }
        };

        MaterialParameter baseColor;
        float baseMetallic = 0.f;
        float baseRoughness = 1.f;
        float baseReflectance = 0.5f;
        float baseClearCoat = 0.f;
        float baseClearCoatRoughness = 0.f;
        float baseAnisotropy = 0.f;

        std::shared_ptr<cloudViewer::geometry::Image> albedo;
        std::shared_ptr<cloudViewer::geometry::Image> normalMap;
        std::shared_ptr<cloudViewer::geometry::Image> ambientOcclusion;
        std::shared_ptr<cloudViewer::geometry::Image> metallic;
        std::shared_ptr<cloudViewer::geometry::Image> roughness;
        std::shared_ptr<cloudViewer::geometry::Image> reflectance;
        std::shared_ptr<cloudViewer::geometry::Image> clearCoat;
        std::shared_ptr<cloudViewer::geometry::Image> clearCoatRoughness;
        std::shared_ptr<cloudViewer::geometry::Image> anisotropy;

        std::unordered_map<std::string, MaterialParameter> floatParameters;
        std::unordered_map<std::string, cloudViewer::geometry::Image>
                additionalMaps;
    };

    std::vector<std::pair<std::string, Material>> materials_;

    /// List of material ids.
    std::vector<int> triangle_material_ids_;
    /// Textures of the image.
    std::vector<cloudViewer::geometry::Image> textures_;

    /// Returns `true` if the mesh contains adjacency normals.
    bool hasAdjacencyList() const {
        return getVerticeSize() > 0 &&
               adjacency_list_.size() == getVerticeSize();
    }

    inline bool hasTriangleUvs() const {
        return hasTriangles() && triangle_uvs_.size() == 3 * size();
    }

    bool hasTriangleMaterialIds() const {
        return hasTriangles() && triangle_material_ids_.size() == size();
    }

    /// Returns `true` if the mesh has texture.
    bool hasEigenTextures() const {
        bool is_all_texture_valid = std::accumulate(
                textures_.begin(), textures_.end(), true,
                [](bool a, const cloudViewer::geometry::Image& b) {
                    return a && !b.IsEmpty();
                });
        return !textures_.empty() && is_all_texture_valid;
    }

    inline virtual bool IsEmpty() const override {
        return !HasVertices() || !hasTriangles();
    }

    virtual Eigen::Vector3d GetMinBound() const override;
    virtual Eigen::Vector3d GetMaxBound() const override;
    virtual Eigen::Vector3d GetCenter() const override;
    virtual ccBBox GetAxisAlignedBoundingBox() const override;
    virtual ecvOrientedBBox GetOrientedBoundingBox() const override;
    virtual ccMesh& Transform(const Eigen::Matrix4d& transformation) override;
    virtual ccMesh& Translate(const Eigen::Vector3d& translation,
                              bool relative = true) override;
    virtual ccMesh& Scale(const double s,
                          const Eigen::Vector3d& center) override;
    virtual ccMesh& Rotate(const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& center) override;

    /// \brief Assigns each vertex in the ccMesh the same color
    ///
    /// \param color RGB colors of vertices.
    ccMesh& PaintUniformColor(const Eigen::Vector3d& color);

    std::tuple<std::shared_ptr<ccMesh>, std::vector<size_t>> ComputeConvexHull()
            const;

    /// \brief Function that computes for each edge in the triangle mesh and
    /// passed as parameter edges_to_vertices the cot weight.
    ///
    /// \param edges_to_vertices map from edge to vector of neighbouring
    /// vertices.
    /// \param min_weight minimum weight returned. Weights smaller than this
    /// get clamped.
    /// \return cot weight per edge.
    std::unordered_map<Eigen::Vector2i,
                       double,
                       cloudViewer::utility::hash_eigen<Eigen::Vector2i>>
    ComputeEdgeWeightsCot(
            const std::unordered_map<
                    Eigen::Vector2i,
                    std::vector<int>,
                    cloudViewer::utility::hash_eigen<Eigen::Vector2i>>&
                    edges_to_vertices,
            double min_weight = std::numeric_limits<double>::lowest()) const;

    /// \brief Function to compute triangle normals, usually called before
    /// rendering.
    ccMesh& ComputeTriangleNormals(bool normalized = true);

    /// \brief Function to compute vertex normals, usually called before
    /// rendering.
    ccMesh& ComputeVertexNormals(bool normalized = true);

    /// Normalize both triangle normals and vertex normals to length 1.
    /// Normalize point normals to length 1.
    ccMesh& NormalizeNormals();

    /// \brief Function to compute adjacency list, call before adjacency list is
    ccMesh& ComputeAdjacencyList();

    /// \brief Function that removes duplicated verties, i.e., vertices that
    /// have identical coordinates.
    ccMesh& RemoveDuplicatedVertices();

    /// \brief Function that removes duplicated triangles, i.e., removes
    /// triangles that reference the same three vertices, independent of their
    /// order.
    ccMesh& RemoveDuplicatedTriangles();

    /// \brief This function removes vertices from the triangle mesh that are
    /// not referenced in any triangle of the mesh.
    ccMesh& RemoveUnreferencedVertices();

    /// \brief Function that removes degenerate triangles, i.e., triangles that
    /// reference a single vertex multiple times in a single triangle.
    ///
    /// They are usually the product of removing duplicated vertices.
    ccMesh& RemoveDegenerateTriangles();

    /// \brief Function that removes all non-manifold edges, by successively
    /// deleting triangles with the smallest surface area adjacent to the
    /// non-manifold edge until the number of adjacent triangles to the edge is
    /// `<= 2`.
    ccMesh& RemoveNonManifoldEdges();

    /// \brief Function that will merge close by vertices to a single one.
    /// The vertex position, normal and color will be the average of the
    /// vertices.
    ///
    /// \param eps defines the maximum distance of close by vertices.
    /// This function might help to close triangle soups.
    ccMesh& MergeCloseVertices(double eps);

    /// \brief Function to sharpen triangle mesh.
    ///
    /// The output value ($v_o$) is the input value ($v_i$) plus strength times
    /// the input value minus the sum of he adjacent values. $v_o = v_i x
    /// strength (v_i * |N| - \sum_{n \in N} v_n)$.
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    /// \param strength - The strength of the filter.
    std::shared_ptr<ccMesh> FilterSharpen(
            int number_of_iterations,
            double strength,
            FilterScope scope = FilterScope::All) const;

    /// \brief Function to smooth triangle mesh with simple neighbour average.
    ///
    /// $v_o = \frac{v_i + \sum_{n \in N} v_n)}{|N| + 1}$, with $v_i$
    /// being the input value, $v_o$ the output value, and $N$ is the
    /// set of adjacent neighbours.
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    std::shared_ptr<ccMesh> FilterSmoothSimple(
            int number_of_iterations,
            FilterScope scope = FilterScope::All) const;

    /// \brief Function to smooth triangle mesh using Laplacian.
    ///
    /// $v_o = v_i \cdot \lambda (sum_{n \in N} w_n v_n - v_i)$,
    /// with $v_i$ being the input value, $v_o$ the output value, $N$ is the
    /// set of adjacent neighbours, $w_n$ is the weighting of the neighbour
    /// based on the inverse distance (closer neighbours have higher weight),
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    /// \param lambda is the smoothing parameter.
    std::shared_ptr<ccMesh> FilterSmoothLaplacian(
            int number_of_iterations,
            double lambda,
            FilterScope scope = FilterScope::All) const;

    /// \brief Function to smooth triangle mesh using method of Taubin,
    /// "Curve and Surface Smoothing Without Shrinkage", 1995.
    /// Applies in each iteration two times FilterSmoothLaplacian, first
    /// with lambda and second with mu as smoothing parameter.
    /// This method avoids shrinkage of the triangle mesh.
    ///
    /// \param number_of_iterations defines the number of repetitions
    /// of this operation.
    /// \param lambda is the filter parameter
    /// \param mu is the filter parameter
    std::shared_ptr<ccMesh> FilterSmoothTaubin(
            int number_of_iterations,
            double lambda = 0.5,
            double mu = -0.53,
            FilterScope scope = FilterScope::All) const;

    /// Function that computes the Euler-Poincaré characteristic, i.e.,
    /// V + F - E, where V is the number of vertices, F is the number
    /// of triangles, and E is the number of edges.
    int EulerPoincareCharacteristic() const;

    /// Function that returns the non-manifold edges of the triangle mesh.
    /// If \param allow_boundary_edges is set to false, than also boundary
    /// edges are returned
    std::vector<Eigen::Vector2i> GetNonManifoldEdges(
            bool allow_boundary_edges = true) const;

    /// Function that checks if the given triangle mesh is edge-manifold.
    /// A mesh is edge­-manifold if each edge is bounding either one or two
    /// triangles. If allow_boundary_edges is set to false, then this function
    /// returns false if there exists boundary edges.
    bool IsEdgeManifold(bool allow_boundary_edges = true) const;

    /// Function that returns a list of non-manifold vertex indices.
    /// A vertex is manifold if its star is edge‐manifold and edge‐connected.
    /// (Two or more faces connected only by a vertex and not by an edge.)
    std::vector<int> GetNonManifoldVertices() const;

    /// Function that checks if all vertices in the triangle mesh are manifold.
    /// A vertex is manifold if its star is edge‐manifold and edge‐connected.
    /// (Two or more faces connected only by a vertex and not by an edge.)
    bool IsVertexManifold() const;

    /// Function that returns a list of triangles that are intersecting the
    /// mesh.
    std::vector<Eigen::Vector2i> GetSelfIntersectingTriangles() const;

    /// Function that tests if the triangle mesh is self-intersecting.
    /// Tests each triangle pair for intersection.
    bool IsSelfIntersecting() const;

    /// Function that tests if the bounding boxes of the triangle meshes are
    /// intersecting.
    bool IsBoundingBoxIntersecting(const ccMesh& other) const;

    /// Function that tests if the triangle mesh intersects another triangle
    /// mesh. Tests each triangle against each other triangle.
    bool IsIntersecting(const ccMesh& other) const;

    /// Function that tests if the given triangle mesh is orientable, i.e.
    /// the triangles can oriented in such a way that all normals point
    /// towards the outside.
    bool IsOrientable() const;

    /// Function that tests if the given triangle mesh is watertight by
    /// checking if it is vertex manifold and edge-manifold with no boundary
    /// edges, but not self-intersecting.
    bool IsWatertight() const;

    /// If the mesh is orientable then this function rearranges the
    /// triangles such that all normals point towards the
    /// outside/inside.
    bool OrientTriangles();

    /// Function that returns a map from edges (vertex0, vertex1) to the
    /// triangle indices the given edge belongs to.
    std::unordered_map<Eigen::Vector2i,
                       std::vector<int>,
                       cloudViewer::utility::hash_eigen<Eigen::Vector2i>>
    GetEdgeToTrianglesMap() const;

    /// Function that returns a map from edges (vertex0, vertex1) to the
    /// vertex (vertex2) indices the given edge belongs to.
    std::unordered_map<Eigen::Vector2i,
                       std::vector<int>,
                       cloudViewer::utility::hash_eigen<Eigen::Vector2i>>
    GetEdgeToVerticesMap() const;

    /// Function that computes the area of a mesh triangle identified by the
    /// triangle index
    double GetTriangleArea(size_t triangle_idx) const;

    /// Function that computes the area of a mesh triangle
    static double ComputeTriangleArea(const Eigen::Vector3d& p0,
                                      const Eigen::Vector3d& p1,
                                      const Eigen::Vector3d& p2);

    static inline Eigen::Vector3i GetEigneOrderedTriangle(int vidx0,
                                                          int vidx1,
                                                          int vidx2) {
        if (vidx0 > vidx2) {
            std::swap(vidx0, vidx2);
        }
        if (vidx0 > vidx1) {
            std::swap(vidx0, vidx1);
        }
        if (vidx1 > vidx2) {
            std::swap(vidx1, vidx2);
        }
        return Eigen::Vector3i(vidx0, vidx1, vidx2);
    }

    /// Function that computes the surface area of the mesh, i.e. the sum of
    /// the individual triangle surfaces.
    double GetSurfaceArea() const;

    /// Function that computes the surface area of the mesh, i.e. the sum of
    /// the individual triangle surfaces.
    double GetSurfaceArea(std::vector<double>& triangle_areas) const;

    /// Function that computes the volume of the mesh, under the condition
    /// that it is watertight and orientable. See Zhang and Chen, "Efficient
    /// feature extraction for 2D/3D objects in mesh representation", 2001.
    double GetVolume() const;

    /// Function that computes the plane equation from the three points.
    /// If the three points are co-linear, then this function returns the
    /// invalid plane (0, 0, 0, 0).
    static Eigen::Vector4d ComputeTrianglePlane(const Eigen::Vector3d& p0,
                                                const Eigen::Vector3d& p1,
                                                const Eigen::Vector3d& p2);

    /// Function that computes the plane equation of a mesh triangle identified
    /// by the triangle index.
    Eigen::Vector4d GetTrianglePlane(size_t triangle_idx) const;

    /// Helper function to get an edge with ordered vertex indices.
    static inline Eigen::Vector2i GetOrderedEdge(int vidx0, int vidx1) {
        return Eigen::Vector2i(std::min(vidx0, vidx1), std::max(vidx0, vidx1));
    }

    /// Function to sample \param number_of_points points uniformly from the
    /// mesh.
    std::shared_ptr<ccPointCloud> SamplePointsUniformlyImpl(
            size_t number_of_points,
            std::vector<double>& triangle_areas,
            double surface_area,
            bool use_triangle_normal,
            int seed);

    /// Function to sample \param number_of_points points uniformly from the
    /// mesh. \param use_triangle_normal Set to true to assign the triangle
    /// normals to the returned points instead of the interpolated vertex
    /// normals. The triangle normals will be computed and added to the mesh
    /// if necessary. \param seed Sets the seed value used in the random
    /// generator, set to -1 to use a random seed value with each function
    /// call.
    std::shared_ptr<ccPointCloud> SamplePointsUniformly(
            size_t number_of_points,
            bool use_triangle_normal = false,
            int seed = -1);

    /// Function to sample \p number_of_points points (blue noise).
    /// Based on the method presented in Yuksel, "Sample Elimination for
    /// Generating Poisson Disk Sample Sets", EUROGRAPHICS, 2015 The
    /// PointCloud \p pcl_init is used for sample elimination if given,
    /// otherwise a PointCloud is first uniformly sampled with \p
    /// init_number_of_points x \p number_of_points number of points. \p
    /// use_triangle_normal Set to true to assign the triangle normals to
    /// the returned points instead of the interpolated vertex normals. The
    /// triangle normals will be computed and added to the mesh if
    /// necessary. \p seed Sets the seed value used in the random generator,
    /// set to -1 to use a random seed value with each function call.
    std::shared_ptr<ccPointCloud> SamplePointsPoissonDisk(
            size_t number_of_points,
            double init_factor = 5,
            const std::shared_ptr<ccPointCloud> pcl_init = nullptr,
            bool use_triangle_normal = false,
            int seed = -1);

    /// Function to subdivide triangle mesh using the simple midpoint algorithm.
    /// Each triangle is subdivided into four triangles per iteration and the
    /// new vertices lie on the midpoint of the triangle edges.
    /// \param number_of_iterations defines a single iteration splits each
    /// triangle into four triangles that cover the same surface.
    std::shared_ptr<ccMesh> SubdivideMidpoint(int number_of_iterations) const;

    /// Function to subdivide triangle mesh using Loop's scheme.
    /// Cf. Charles T. Loop, "Smooth subdivision surfaces based on triangles",
    /// 1987. Each triangle is subdivided into four triangles per iteration.
    /// \param number_of_iterations defines a single iteration splits each
    /// triangle into four triangles that cover the same surface.
    std::shared_ptr<ccMesh> SubdivideLoop(int number_of_iterations) const;

    /// Function to simplify mesh using Vertex Clustering.
    /// The result can be a non-manifold mesh.
    /// \param voxel_size - The size of the voxel within vertices are pooled.
    /// \param contraction - Method to aggregate vertex information. Average
    /// computes a simple average, Quadric minimizes the distance to the
    /// adjacent planes.
    std::shared_ptr<ccMesh> SimplifyVertexClustering(
            double voxel_size,
            SimplificationContraction contraction =
                    SimplificationContraction::Average) const;

    /// Function to simplify mesh using Quadric Error Metric Decimation by
    /// Garland and Heckbert.
    /// \param target_number_of_triangles defines the number of triangles
    /// that the simplified mesh should have. It is not guaranteed that this
    /// number will be reached. \param maximum_error defines the maximum
    /// error where a vertex is allowed to be merged \param boundary_weight
    /// a weight applied to edge vertices used to preserve boundaries
    std::shared_ptr<ccMesh> SimplifyQuadricDecimation(
            int target_number_of_triangles,
            double maximum_error = std::numeric_limits<double>::infinity(),
            double boundary_weight = 1.0) const;

    /// Function to select points from \p input ccMesh into
    /// output ccMesh
    /// Vertices with indices in \p indices are selected.
    /// \param indices defines Indices of vertices to be selected.
    /// \param cleanup If true it automatically calls
    /// ccMesh::RemoveDuplicatedVertices,
    /// ccMesh::RemoveDuplicatedTriangles,
    /// ccMesh::RemoveUnreferencedVertices, and
    /// ccMesh::RemoveDegenerateTriangles
    std::shared_ptr<ccMesh> SelectByIndex(const std::vector<size_t>& indices,
                                          bool cleanup = true) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \param bbox are
    /// clipped.
    /// \param bbox defines the input Axis Aligned Bounding Box.
    std::shared_ptr<ccMesh> Crop(const ccBBox& bbox) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \param bbox are
    /// clipped.
    /// \param bbox defines the input Oriented Bounding Box.
    std::shared_ptr<ccMesh> Crop(const ecvOrientedBBox& bbox) const;

    /// \brief Function that clusters connected triangles, i.e., triangles that
    /// are connected via edges are assigned the same cluster index.
    ///
    /// \return A vector that contains the cluster index per
    /// triangle, a second vector contains the number of triangles per
    /// cluster, and a third vector contains the surface area per cluster.
    std::tuple<std::vector<int>, std::vector<size_t>, std::vector<double>>
    ClusterConnectedTriangles() const;

    /// \brief This function removes the triangles with index in
    /// \p triangle_indices. Call \ref RemoveUnreferencedVertices to clean up
    /// vertices afterwards.
    ///
    /// \param triangle_indices Indices of the triangles that should be
    /// removed.
    void RemoveTrianglesByIndex(const std::vector<size_t>& triangle_indices);

    /// \brief This function removes the triangles that are masked in
    /// \p triangle_mask. Call \ref RemoveUnreferencedVertices to clean up
    /// vertices afterwards.
    ///
    /// \param triangle_mask Mask of triangles that should be removed.
    /// Should have same size as \ref triangles_.
    void RemoveTrianglesByMask(const std::vector<bool>& triangle_mask);

    /// \brief This function removes the vertices with index in
    /// \p vertex_indices. Note that also all triangles associated with the
    /// vertices are removeds.
    ///
    /// \param triangle_indices Indices of the triangles that should be
    /// removed.
    void RemoveVerticesByIndex(const std::vector<size_t>& vertex_indices);

    /// \brief This function removes the vertices that are masked in
    /// \p vertex_mask. Note that also all triangles associated with the
    /// vertices are removed..
    ///
    /// \param vertex_mask Mask of vertices that should be removed.
    /// Should have same size as \ref vertices_.
    void RemoveVerticesByMask(const std::vector<bool>& vertex_mask);

    /// \brief This function deforms the mesh using the method by
    /// Sorkine and Alexa, "As-Rigid-As-Possible Surface Modeling", 2007.
    ///
    /// \param constraint_vertex_indices Indices of the triangle vertices
    /// that should be constrained by the vertex positions in
    /// constraint_vertex_positions.
    /// \param constraint_vertex_positions Vertex positions used for the
    /// constraints.
    /// \param max_iter maximum number of iterations to minimize energy
    /// functional.
    /// \param energy energy model that should be optimized
    /// \param smoothed_alpha alpha parameter of the smoothed ARAP model
    /// \return The deformed ccMesh
    std::shared_ptr<ccMesh> DeformAsRigidAsPossible(
            const std::vector<int>& constraint_vertex_indices,
            const std::vector<Eigen::Vector3d>& constraint_vertex_positions,
            size_t max_iter,
            DeformAsRigidAsPossibleEnergy energy =
                    DeformAsRigidAsPossibleEnergy::Spokes,
            double smoothed_alpha = 0.01) const;

    /// \brief Alpha shapes are a generalization of the convex hull. With
    /// decreasing alpha value the shape schrinks and creates cavities.
    /// See Edelsbrunner and Muecke, "Three-Dimensional Alpha Shapes", 1994.
    /// \param pcd PointCloud for what the alpha shape should be computed.
    /// \param alpha parameter to controll the shape. A very big value will
    /// give a shape close to the convex hull.
    /// \param tetra_mesh If not a nullptr, than uses this to construct the
    /// alpha shape. Otherwise, ComputeDelaunayTetrahedralization is called.
    /// \param pt_map Optional map from tetra_mesh vertex indices to pcd
    /// points.
    /// \return ccMesh of the alpha shape.
    static std::shared_ptr<ccMesh> CreateFromPointCloudAlphaShape(
            const ccPointCloud& pcd,
            double alpha,
            std::shared_ptr<cloudViewer::geometry::TetraMesh> tetra_mesh =
                    nullptr,
            std::vector<size_t>* pt_map = nullptr);

    /// Function that computes a triangle mesh from a oriented PointCloud \param
    /// pcd. This implements the Ball Pivoting algorithm proposed in F.
    /// Bernardini et al., "The ball-pivoting algorithm for surface
    /// reconstruction", 1999. The implementation is also based on the
    /// algorithms outlined in Digne, "An Analysis and Implementation of a
    /// Parallel Ball Pivoting Algorithm", 2014. The surface reconstruction is
    /// done by rolling a ball with a given radius (cf. \param radii) over the
    /// point cloud, whenever the ball touches three points a triangle is
    /// created.
    /// \param pcd defines the PointCloud from which the ccMesh surface is
    /// reconstructed. Has to contain normals. \param radii defines the radii of
    /// the ball that are used for the surface reconstruction.
    static std::shared_ptr<ccMesh> CreateFromPointCloudBallPivoting(
            const ccPointCloud& pcd, const std::vector<double>& radii);

    /// \brief Function that computes a triangle mesh from a oriented PointCloud
    /// pcd. This implements the Screened Poisson Reconstruction proposed in
    /// Kazhdan and Hoppe, "Screened Poisson Surface Reconstruction", 2013.
    /// This function uses the original implementation by Kazhdan. See
    /// https://github.com/mkazhdan/PoissonRecon
    ///
    /// \param pcd PointCloud with normals and optionally colors.
    /// \param depth Maximum depth of the tree that will be used for surface
    /// reconstruction. Running at depth d corresponds to solving on a grid
    /// whose resolution is no larger than 2^d x 2^d x 2^d. Note that since the
    /// reconstructor adapts the octree to the sampling density, the specified
    /// reconstruction depth is only an upper bound.
    /// \param width Specifies the
    /// target width of the finest level octree cells. This parameter is ignored
    /// if depth is specified.
    /// \param scale Specifies the ratio between the
    /// diameter of the cube used for reconstruction and the diameter of the
    /// samples' bounding cube.
    /// \param linear_fit If true, the reconstructor use
    /// linear interpolation to estimate the positions of iso-vertices.
    /// \param point_weight The importance that interpolation of the point
    /// samples is given in the formulation of the screened Poisson equation.
    /// The results of the original (unscreened) Poisson Reconstruction
    /// can be obtained by setting this value to 0
    /// \param samples_per_node The minimum number of sample points that should
    /// fall within an octree node as the octree construction is adapted to
    /// sampling density. This parameter specifies the minimum number of points
    /// that should fall within an octree node. For noise-free samples, small
    /// values in the range [1.0 - 5.0] can be used. For more noisy samples,
    /// larger values in the range [15.0 - 20.0] may be needed to provide a
    /// smoother, noise-reduced, reconstruction. \param boundary_type Boundary
    /// type for the finite elements \param n_threads Number of threads used for
    /// reconstruction. Set to -1
    /// to automatically determine it.
    /// \return The estimated ccMesh, and per vertex densitie values that
    /// can be used to to trim the mesh.
    static std::tuple<std::shared_ptr<ccMesh>, std::vector<double>>
    CreateFromPointCloudPoisson(const ccPointCloud& pcd,
                                size_t depth = 8,
                                size_t width = 0,
                                float scale = 1.1f,
                                bool linear_fit = false,
                                float point_weight = 2.f,
                                float samples_per_node = 1.5f,
                                int boundary_type = 2 /*BOUNDARY_NEUMANN*/,
                                int n_threads = -1);

    /// Factory function to create a tetrahedron mesh (trianglemeshfactory.cpp).
    /// the mesh centroid will be at (0,0,0) and \p radius defines the
    /// distance from the center to the mesh vertices.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<ccMesh> CreateTetrahedron(
            double radius = 1.0, bool create_uv_map = false);

    /// Factory function to create an octahedron mesh (trianglemeshfactory.cpp).
    /// the mesh centroid will be at (0,0,0) and \p radius defines the
    /// distance from the center to the mesh vertices.
    /// \param radius defines the distance from centroid to mesh vetices.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<ccMesh> CreateOctahedron(double radius = 1.0,
                                                    bool create_uv_map = false);

    /// Factory function to create an icosahedron mesh
    /// (trianglemeshfactory.cpp). The mesh centroid will be at (0,0,0) and
    /// \param radius defines the distance from the center to the mesh vertices.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<ccMesh> CreateIcosahedron(
            double radius = 1.0, bool create_uv_map = false);

    /// Factory function to create a plane mesh (TriangleMeshFactory.cpp)
    /// The left bottom corner on the front will be placed at (0, 0, 0).
    /// \param width is x-directional length.
    /// \param height is y-directional length.
    /// \param create_uv_map add default UV map to the shape.
    static std::shared_ptr<ccMesh> CreatePlane(double width = 1.0,
                                               double height = 1.0,
                                               bool create_uv_map = false);

    /// Factory function to create a box mesh (TriangleMeshFactory.cpp)
    /// The left bottom corner on the front will be placed at (0, 0, 0).
    /// \param width is x-directional length.
    /// \param height is y-directional length.
    /// \param depth is z-directional length.
    /// \param create_uv_map add default UV map to the shape.
    /// \param map_texture_to_each_face if true, maps the entire texture image
    /// to each face. If false, sets the default uv map to the mesh.
    static std::shared_ptr<ccMesh> CreateBox(
            double width = 1.0,
            double height = 1.0,
            double depth = 1.0,
            bool create_uv_map = false,
            bool map_texture_to_each_face = false);

    /// Factory function to create a sphere mesh (TriangleMeshFactory.cpp)
    /// The sphere with radius will be centered at (0, 0, 0).
    /// Its axis is aligned with z-axis.
    /// \param radius defines radius of the sphere.
    /// \param resolution defines the resolution of the sphere. The longitudes
    /// will be split into resolution segments (i.e. there are resolution + 1
    /// latitude lines including the north and south pole). The latitudes will
    /// be split into `2 * resolution segments (i.e. there are 2 * resolution
    /// longitude lines.)
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<ccMesh> CreateSphere(double radius = 1.0,
                                                int resolution = 20,
                                                bool create_uv_map = false);

    /// Factory function to create a cylinder mesh (TriangleMeshFactory.cpp)
    /// The axis of the cylinder will be from (0, 0, -height/2) to (0, 0,
    /// height/2). The circle with radius will be split into
    /// resolution segments. The height will be split into split
    /// segments.
    /// \param radius defines the radius of the cylinder.
    /// \param height defines the height of the cylinder.
    /// \param resolution defines that the circle will be split into resolution
    /// segments \param split defines that the height will be split into split
    /// segments.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<ccMesh> CreateCylinder(double radius = 1.0,
                                                  double height = 2.0,
                                                  int resolution = 20,
                                                  int split = 4,
                                                  bool create_uv_map = false);

    /// Factory function to create a cone mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone will be from (0, 0, 0) to (0, 0, height).
    /// The circle with radius will be split into resolution
    /// segments. The height will be split into split segments.
    /// \param radius defines the radius of the cone.
    /// \param height defines the height of the cone.
    /// \param resolution defines that the circle will be split into resolution
    /// segments \param split defines that the height will be split into split
    /// segments.
    /// \param create_uv_map add default UV map to the mesh.
    static std::shared_ptr<ccMesh> CreateCone(double radius = 1.0,
                                              double height = 2.0,
                                              int resolution = 20,
                                              int split = 1,
                                              bool create_uv_map = false);

    /// Factory function to create a torus mesh (TriangleMeshFactory.cpp)
    /// The torus will be centered at (0, 0, 0) and a radius of
    /// torus_radius. The tube of the torus will have a radius of
    /// tube_radius. The number of segments in radial and tubular direction are
    /// radial_resolution and tubular_resolution respectively.
    /// \param torus_radius defines the radius from the center of the torus to
    /// the center of the tube. \param tube_radius defines the radius of the
    /// torus tube. \param radial_resolution defines the he number of segments
    /// along the radial direction. \param tubular_resolution defines the number
    /// of segments along the tubular direction.
    static std::shared_ptr<ccMesh> CreateTorus(double torus_radius = 1.0,
                                               double tube_radius = 0.5,
                                               int radial_resolution = 30,
                                               int tubular_resolution = 20);

    /// Factory function to create an arrow mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone with cone_radius will be along the z-axis.
    /// The cylinder with cylinder_radius is from
    /// (0, 0, 0) to (0, 0, cylinder_height), and
    /// the cone is from (0, 0, cylinder_height)
    /// to (0, 0, cylinder_height + cone_height).
    /// The cone will be split into resolution segments.
    /// The cylinder_height will be split into cylinder_split
    /// segments. The cone_height will be split into cone_split
    /// segments.
    /// \param cylinder_radius defines the radius of the cylinder.
    /// \param cone_radius defines the radius of the cone.
    /// \param cylinder_height defines the height of the cylinder. The cylinder
    /// is from (0, 0, 0) to (0, 0, cylinder_height) \param cone_height defines
    /// the height of the cone. The axis of the cone will be from (0, 0,
    /// cylinder_height) to (0, 0, cylinder_height + cone_height). \param
    /// resolution defines the cone will be split into resolution segments.
    /// \param cylinder_split defines the cylinder_height will be split into
    /// cylinder_split segments. \param cone_split defines the cone_height will
    /// be split into cone_split segments.
    static std::shared_ptr<ccMesh> CreateArrow(double cylinder_radius = 1.0,
                                               double cone_radius = 1.5,
                                               double cylinder_height = 5.0,
                                               double cone_height = 4.0,
                                               int resolution = 20,
                                               int cylinder_split = 4,
                                               int cone_split = 1);

    /// Factory function to create a coordinate frame mesh
    /// (TriangleMeshFactory.cpp).
    /// arrows respectively. \param size is the length of the axes.
    /// \param size defines the size of the coordinate frame.
    /// \param origin defines the origin of the cooridnate frame.
    static std::shared_ptr<ccMesh> CreateCoordinateFrame(
            double size = 1.0,
            const Eigen::Vector3d& origin = Eigen::Vector3d(0.0, 0.0, 0.0));

    /// Factory function to create a Mobius strip.
    /// \param length_split defines the number of segments along the Mobius
    /// strip.
    /// \param width_split defines the number of segments along the width
    /// of the Mobius strip.\param twists defines the number of twists of the
    /// strip.
    /// \param radius defines the radius of the Mobius strip.
    /// \param flatness controls the height of the strip.
    /// \param width controls the width of the Mobius strip.
    /// \param scale is used to scale the entire Mobius strip.
    static std::shared_ptr<ccMesh> CreateMobius(int length_split = 70,
                                                int width_split = 15,
                                                int twists = 1,
                                                double radius = 1,
                                                double flatness = 1,
                                                double width = 1,
                                                double scale = 1);

protected:
    void FilterSmoothLaplacianHelper(
            std::shared_ptr<ccMesh>& mesh,
            const std::vector<CCVector3>& prev_vertices,
            const std::vector<CCVector3>& prev_vertex_normals,
            const ColorsTableType& prev_vertex_colors,
            const std::vector<std::unordered_set<int>>& adjacency_list,
            double lambda,
            bool filter_vertex,
            bool filter_normal,
            bool filter_color) const;

protected:
    // inherited from ccHObject
    void drawMeOnly(CC_DRAW_CONTEXT& context) override;
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
    void applyGLTransformation(const ccGLMatrix& trans) override;
    void onUpdateOf(ccHObject* obj) override;
    void onDeletionOf(const ccHObject* obj) override;

    //! Same as other 'computeInterpolationWeights' method with a set of 3
    //! vertices indexes
    void computeInterpolationWeights(
            const cloudViewer::VerticesIndexes& vertIndexes,
            const CCVector3& P,
            CCVector3d& weights) const;
    //! Same as other 'interpolateNormals' method with a set of 3 vertices
    //! indexes
    bool interpolateNormals(const cloudViewer::VerticesIndexes& vertIndexes,
                            const CCVector3d& w,
                            CCVector3& N,
                            const Tuple3i* triNormIndexes = nullptr);
    //! Same as other 'interpolateColors' method with a set of 3 vertices
    //! indexes
    bool interpolateColors(const cloudViewer::VerticesIndexes& vertIndexes,
                           const CCVector3& P,
                           ecvColor::Rgb& C);

    //! Used internally by 'subdivide'
    bool pushSubdivide(/*PointCoordinateType maxArea, */ unsigned indexA,
                       unsigned indexB,
                       unsigned indexC);

    /*** EXTENDED CALL SCRIPTS (FOR CC_SUB_MESHES) ***/

    // 0 parameter
#define ccMesh_extended_call0(baseName, recursiveName)        \
    inline virtual void recursiveName() {                     \
        baseName();                                           \
        for (Container::iterator it = m_children.begin();     \
             it != m_children.end(); ++it)                    \
            if ((*it)->isA(CV_TYPES::SUB_MESH))               \
                static_cast<ccGenericMesh*>(*it)->baseName(); \
    }

    // 1 parameter
#define ccMesh_extended_call1(baseName, param1Type, recursiveName) \
    inline virtual void recursiveName(param1Type p) {              \
        baseName(p);                                               \
        for (Container::iterator it = m_children.begin();          \
             it != m_children.end(); ++it)                         \
            if ((*it)->isA(CV_TYPES::SUB_MESH))                    \
                static_cast<ccGenericMesh*>(*it)->baseName(p);     \
    }

    // recursive equivalents of some of ccGenericMesh methods (applied to
    // sub-meshes as well)
    ccMesh_extended_call1(showNormals, bool, showNormals_extended)

            //! associated cloud (vertices)
            ccGenericPointCloud* m_associatedCloud;

    //! Per-triangle normals
    NormsIndexesTableType* m_triNormals;

    //! Texture coordinates
    TextureCoordsContainer* m_texCoords;

    //! Materials
    ccMaterialSet* m_materials;

    //! Triangles' vertices indexes (3 per triangle)
    triangleIndexesContainer* m_triVertIndexes;

    //! Iterator on the list of triangles
    unsigned m_globalIterator;
    //! Dump triangle structure to transmit temporary data
    cloudViewer::SimpleRefTriangle m_currentTriangle;

    //! Bounding-box
    ccBBox m_bBox;

    //! Per-triangle material indexes
    triangleMaterialIndexesSet* m_triMtlIndexes;

    //! Set of triplets of indexes referring to mesh texture coordinates
    using triangleTexCoordIndexesSet = ccArray<Tuple3i, 3, int>;
    //! Mesh tex coords indexes (per-triangle)
    triangleTexCoordIndexesSet* m_texCoordIndexes;

    //! Set of triplets of indexes referring to mesh normals
    using triangleNormalsIndexesSet = ccArray<Tuple3i, 3, int>;
    //! Mesh normals indexes (per-triangle)
    triangleNormalsIndexesSet* m_triNormalIndexes;
};
