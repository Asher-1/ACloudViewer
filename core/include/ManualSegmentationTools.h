//##########################################################################
//#                                                                        #
//#                               cloudViewer                              #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef MANUAL_SEGMENTATION_TOOLS_HEADER
#define MANUAL_SEGMENTATION_TOOLS_HEADER

// Local
#include "Neighbourhood.h"

namespace cloudViewer {

class GenericIndexedCloud;
class GenericIndexedCloudPersist;
class GenericIndexedMesh;
class GenericProgressCallback;
class ReferenceCloud;
class SimpleMesh;
class Polyline;

//! Manual segmentation algorithms (inside/outside a polyline, etc.)
class CV_CORE_LIB_API ManualSegmentationTools : public CVToolbox {
public:
    //! Extracts the points that fall inside/outside of a 2D polyline once
    //! projected on the screen
    /** The camera parameters of the screen must be transmitted to this method,
            as well as the polyline (generally drawn on the screen by a user)
            expressed in the screen coordinates.
            \param aCloud the cloud to segment
            \param poly the polyline
            \param keepInside if true (resp. false), the points falling inside
    (resp. outside) the polyline will be extracted \param viewMat the optional
    4x4 visualization matrix (OpenGL style) \return a cloud structure containing
    references to the extracted points (references to - no duplication)
    **/
    static ReferenceCloud* segment(GenericIndexedCloudPersist* aCloud,
                                   const Polyline* poly,
                                   bool keepInside,
                                   const float* viewMat = nullptr);

    //! Selects the points which associated scalar value fall inside or outside
    //! a specified interval
    /** \warning: be sure to activate an OUTPUT scalar field on the input cloud
            \param cloud the cloud to segment
            \param minDist the lower boundary
            \param maxDist the upper boundary
            \param outside whether to select the points inside or outside
            \return a new cloud structure containing the extracted points
    (references to - no duplication)
    **/
    static ReferenceCloud* segment(GenericIndexedCloudPersist* cloud,
                                   ScalarType minDist,
                                   ScalarType maxDist,
                                   bool outside = false);

    //! Selects the points which associated scalar value fall inside or outside
    //! a specified interval
    /** \warning: be sure to activate an OUTPUT scalar field on the input cloud
            \param cloud the cloud to segment
            \param values scalar values
            \param outside whether to select the points inside or outside
            \return a new cloud structure containing the extracted points
    (references to - no duplication)
    **/
    static ReferenceCloud* segment(GenericIndexedCloudPersist* cloud,
                                   std::vector<ScalarType> values,
                                   bool outside = false);

    //! Tests if a point is inside a polygon (2D)
    /** \param P a 2D point
            \param polyVertices polygon vertices (considered as ordered 2D
    poyline vertices) \return true if P is inside poly
    **/
    static bool isPointInsidePoly(const CCVector2& P,
                                  const GenericIndexedCloud* polyVertices);

    //! Tests if a point is inside a polygon (2D)
    /** \param P a 2D point
            \param polyVertices polygon vertices (considered as ordered 2D
    poyline vertices) \return true if P is inside poly
    **/
    static bool isPointInsidePoly(const CCVector2& P,
                                  const std::vector<CCVector2>& polyVertices);

    //! Segments a mesh knowing which vertices should be kept or not
    /** This method takes as input a set of vertex indexes and creates a new
    mesh composed either of:
            - the triangles that have exactly those points as vertices
    (useSelectedVertices = true)
            - or all the triangles for which no vertex is part of this subset
    (useSelectedVertices = false).

            \warning No re-triangulation on the border will occur.

            \param mesh a mesh
            \param selectedVertexIndexes the indexes of selected vertices
            \param useSelectedVertices specifies if the points corresponding to
    the input indexes should be the new mesh vertices, or the opposite \param
    progressCb the client application can get some notification of the process
    progress through this callback mechanism (see GenericProgressCallback)
            \param destCloud optionally, a cloud object can be specified to be
    associated to the new created mesh object, instead of the cloud associated
    to the ReferenceCloud "pointsIndexes" \param indexShift optionally, a shift
    can be added to all vertex indexes of the new mesh \param triangleIndexMap
    optionally, a vector can be filled to record the new indexes of the
    triangles in the destination mesh (or -1 if they are discarded) \return a
    new mesh structure, or 0 if something went wrong
    **/
    static GenericIndexedMesh* segmentMesh(
            GenericIndexedMesh* mesh,
            ReferenceCloud* selectedVertexIndexes,
            bool useSelectedVertices,
            GenericProgressCallback* progressCb = nullptr,
            GenericIndexedCloud* destCloud = nullptr,
            unsigned indexShift = 0,
            std::vector<int>* triangleIndexMap = nullptr);

    //! Input/output parameters for the segmentMeshWithAAPlane method
    struct MeshCutterParams {
        SimpleMesh* insideMesh;
        SimpleMesh* outsideMesh;
        bool generateOutsideMesh;
        double epsilon;
        // for infinite plane intersection
        unsigned char planeOrthoDim;
        double planeCoord;
        // for box intersection
        CCVector3d bbMin, bbMax;
        // for the reprojection of triangle features
        bool trackOrigIndexes;
        std::vector<unsigned> origTriIndexesMapInside;
        std::vector<unsigned> origTriIndexesMapOutside;

        MeshCutterParams()
            : insideMesh(nullptr),
              outsideMesh(nullptr),
              generateOutsideMesh(false),
              epsilon(ZERO_TOLERANCE_D),
              planeOrthoDim(0),
              planeCoord(0),
              bbMin(0, 0, 0),
              bbMax(0, 0, 0),
              trackOrigIndexes(false) {}
    };

    static bool segmentMeshWithAAPlane(
            GenericIndexedMesh* mesh,
            GenericIndexedCloudPersist* vertices,
            MeshCutterParams& ioParams,
            GenericProgressCallback* progressCb = nullptr);

    static bool segmentMeshWithAABox(
            GenericIndexedMesh* mesh,
            GenericIndexedCloudPersist* vertices,
            MeshCutterParams& ioParams,
            GenericProgressCallback* progressCb = nullptr);
};

}  // namespace cloudViewer

#endif  // MANUAL_SEGMENTATION_TOOLS_HEADER
