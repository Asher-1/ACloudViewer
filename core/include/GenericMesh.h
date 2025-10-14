// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef GENERIC_MESH_HEADER
#define GENERIC_MESH_HEADER

#include <functional>

// Local
#include "CVGeom.h"

namespace cloudViewer {

class GenericTriangle;

//! A generic mesh interface for data communication between library and client
//! applications
class CV_CORE_LIB_API GenericMesh {
public:
    GenericMesh() = default;
    //! Default destructor
    virtual ~GenericMesh() = default;

    /// \brief Indicates the method that is used for mesh simplification if
    /// multiple vertices are combined to a single one.
    ///
    /// \param Average indicates that the average position is computed as
    /// output.
    /// \param Quadric indicates that the distance to the adjacent triangle
    /// planes is minimized. Cf. "Simplifying Surfaces with Color and Texture
    /// using Quadric Error Metrics" by Garland and Heckbert.
    enum class SimplificationContraction { Average, Quadric };

    /// \brief Indicates the scope of filter operations.
    ///
    /// \param All indicates that all properties (color, normal,
    /// vertex position) are filtered.
    /// \param Color indicates that only the colors are filtered.
    /// \param Normal indicates that only the normals are filtered.
    /// \param Vertex indicates that only the vertex positions are filtered.
    enum class FilterScope { All, Color, Normal, Vertex };

    /// Energy model that is minimized in the DeformAsRigidAsPossible method.
    /// \param Spokes is the original energy as formulated in
    /// Sorkine and Alexa, "As-Rigid-As-Possible Surface Modeling", 2007.
    /// \param Smoothed adds a rotation smoothing term to the rotations.
    enum class DeformAsRigidAsPossibleEnergy { Spokes, Smoothed };

    //! Generic function to apply to a triangle (used by foreach)
    using genericTriangleAction = std::function<void(GenericTriangle&)>;

    //! Returns the number of triangles
    /**	Virtual method to request the mesh size
            \return the mesh size
    **/
    virtual unsigned size() const = 0;
    virtual bool hasTriangles() const { return size() != 0; }

    //! Fast iteration mechanism
    /**	Virtual method to apply a function to the whole mesh
            \param action function to apply (see
    GenericMesh::genericTriangleAction)
    **/
    virtual void forEach(genericTriangleAction action) = 0;

    //! Returns the mesh bounding-box
    /**	Virtual method to request the mesh bounding-box limits. It is equivalent
    to the bounding-box of the cloud composed of the mesh vertexes. \param bbMin
    lower bounding-box limits (Xmin,Ymin,Zmin) \param bbMax higher bounding-box
    limits (Xmax,Ymax,Zmax)
    **/
    virtual void getBoundingBox(CCVector3& bbMin, CCVector3& bbMax) = 0;

    //! Places the mesh iterator at the beginning
    /**	Virtual method to handle the mesh global iterator
     **/
    virtual void placeIteratorAtBeginning() = 0;

    //! Returns the next triangle (relatively to the global iterator position)
    /**	Virtual method to handle the mesh global iterator.
            Global iterator position should be increased each time
            this method is called. The returned object can be temporary.
            \return a triangle
    **/
    virtual GenericTriangle* _getNextTriangle() = 0;  // temporary
};

}  // namespace cloudViewer

#endif  // GENERIC_MESH_HEADER
