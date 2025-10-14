// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef MESH_ILLUMINATION_HEADER
#define MESH_ILLUMINATION_HEADER

// cloudViewer
#include <GenericCloud.h>
#include <GenericIndexedMesh.h>
#include <GenericMesh.h>
#include <GenericProgressCallback.h>

// Qt
#include <QString>

// System
#include <vector>

//! PCV (Portion de Ciel Visible) algorithm
/** "Ambient Occlusion" in english!
 **/
class PCV {
public:
    //! Simulates global illumination on a cloud (or a mesh) with OpenGL -
    //! shortcut version
    /** Computes per-vertex illumination intensity as a scalar field.
            \param numberOfRays (approxiamate) number of rays to generate
            \param mode360 whether light rays should be generated on the half
    superior sphere (false) or the whole sphere (true) \param vertices vertices
    (eventually corresponding to a mesh - see below) to englight \param mesh
    optional mesh structure associated to the vertices \param meshIsClosed if a
    mesh is passed as argument (see above), specifies if the mesh surface is
    closed (enables optimization) \param width width  of the OpenGL context used
    to simulate illumination \param height height of the OpenGL context used to
    simulate illumination \param progressCb optional progress bar (optional)
            \param entityName entity name (optional)
            \return number of 'light' directions actually used (or a value <0 if
    an error occurred)
    **/
    static int Launch(
            unsigned numberOfRays,
            cloudViewer::GenericCloud* vertices,
            cloudViewer::GenericMesh* mesh = nullptr,
            bool meshIsClosed = false,
            bool mode360 = true,
            unsigned width = 1024,
            unsigned height = 1024,
            cloudViewer::GenericProgressCallback* progressCb = nullptr,
            QString entityName = QString());

    //! Simulates global illumination on a cloud (or a mesh) with OpenGL
    /** Computes per-vertex illumination intensity as a scalar field.
            \param rays light directions that will be used to compute global
    illumination \param vertices vertices (eventually corresponding to a mesh -
    see below) to englight \param mesh optional mesh structure associated to the
    vertices \param meshIsClosed if a mesh is passed as argument (see above),
    specifies if the mesh surface is closed (enables optimization) \param width
    width  of the OpenGL context used to simulate illumination \param height
    height of the OpenGL context used to simulate illumination \param progressCb
    optional progress bar (optional) \param entityName entity name (optional)
            \return success
    **/
    static bool Launch(
            std::vector<CCVector3>& rays,
            cloudViewer::GenericCloud* vertices,
            cloudViewer::GenericMesh* mesh = nullptr,
            bool meshIsClosed = false,
            unsigned width = 1024,
            unsigned height = 1024,
            cloudViewer::GenericProgressCallback* progressCb = nullptr,
            QString entityName = QString());

    //! Generates a given number of rays
    static bool GenerateRays(unsigned numberOfRays,
                             std::vector<CCVector3>& rays,
                             bool mode360 = true);
};

#endif
