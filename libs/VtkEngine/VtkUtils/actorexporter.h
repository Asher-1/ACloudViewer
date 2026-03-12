// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file actorexporter.h
/// @brief Exports VTK actor geometry to file (e.g. STL).

#include <QRunnable>
#include <QString>

#include "qVTK.h"

class vtkActor;
namespace VtkUtils {

/// @class ActorExporter
/// @brief QRunnable that exports a vtkActor to a file.
class QVTK_ENGINE_LIB_API ActorExporter : public QRunnable {
public:
    /// @param actor VTK actor to export
    /// @param file Output file path
    ActorExporter(vtkActor* actor, const QString& file);

    void run();

protected:
    vtkActor* m_actor = nullptr;
    QString m_exportFile;
};

}  // namespace VtkUtils
