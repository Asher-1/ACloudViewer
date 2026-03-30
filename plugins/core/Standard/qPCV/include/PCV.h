// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <GenericCloud.h>
#include <GenericIndexedMesh.h>
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
    //! Simulates global illumination on a cloud (or a mesh) - shortcut version
    /** Computes per-vertex illumination intensity as a scalar field.
        \param numberOfRays (approximate) number of rays to generate
        \param vertices vertices (eventually corresponding to a mesh) to illuminate
        \param mesh optional mesh structure associated to the vertices
        \param meshIsClosed if a mesh is passed, specifies if the mesh surface is closed
        \param mode360 whether light rays should be on the half sphere (false) or whole sphere (true)
        \param width width of the render context used to simulate illumination
        \param height height of the render context used to simulate illumination
        \param progressCb optional progress bar
        \param entityName entity name (optional)
        \return number of 'light' directions actually used (or a value <0 if an error occurred)
    **/
    static int Launch(unsigned numberOfRays,
                      cloudViewer::GenericCloud* vertices,
                      cloudViewer::GenericMesh* mesh = nullptr,
                      bool meshIsClosed = false,
                      bool mode360 = true,
                      unsigned width = 1024,
                      unsigned height = 1024,
                      cloudViewer::GenericProgressCallback* progressCb = nullptr,
                      const QString& entityName = QString());

    //! Simulates global illumination on a cloud (or a mesh)
    /** Computes per-vertex illumination intensity as a scalar field.
        \param rays light directions that will be used to compute global illumination
        \param vertices vertices to illuminate
        \param mesh optional mesh structure associated to the vertices
        \param meshIsClosed if a mesh is passed, specifies if the mesh surface is closed
        \param width width of the render context
        \param height height of the render context
        \param progressCb optional progress bar
        \param entityName entity name (optional)
        \return success
    **/
    static bool Launch(const std::vector<CCVector3d>& rays,
                       cloudViewer::GenericCloud* vertices,
                       cloudViewer::GenericMesh* mesh = nullptr,
                       bool meshIsClosed = false,
                       unsigned width = 1024,
                       unsigned height = 1024,
                       cloudViewer::GenericProgressCallback* progressCb = nullptr,
                       const QString& entityName = QString());

    //! Generates a given number of rays
    static bool GenerateRays(unsigned numberOfRays,
                             std::vector<CCVector3d>& rays,
                             bool mode360 = true);
};
