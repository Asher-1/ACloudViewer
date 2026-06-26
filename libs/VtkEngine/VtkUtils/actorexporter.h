// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file actorexporter.h
/// @brief Exports VTK actor geometry to file (e.g. STL).

#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>

#include <QRunnable>
#include <QString>

#include "qVTK.h"

class vtkActor;

namespace VtkUtils {

/// @class ActorExporter
/// @brief QRunnable that exports a vtkActor to a file.
///
/// Deep-copies the actor's geometry on construction (GUI thread) so the
/// export runs on a pool thread without sharing VTK objects across threads.
class QVTK_ENGINE_LIB_API ActorExporter : public QRunnable {
public:
    /// @param actor VTK actor to export (geometry is deep-copied immediately)
    /// @param file Output file path
    ActorExporter(vtkActor* actor, const QString& file);

    void run();

protected:
    vtkSmartPointer<vtkPolyData> m_polyData;
    vtkSmartPointer<vtkProperty> m_property;
    QString m_exportFile;
};

}  // namespace VtkUtils
