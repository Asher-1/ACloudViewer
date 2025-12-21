// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "qPCL.h"

// VTK
#include <vtkSmartPointer.h>

// QT
#include <QCursor>
#include <QObject>

// Forward declarations
class ecvGenericVisualizer3D;
class vtkRenderWindowInteractor;
class vtkRenderer;
class vtkInteractorStyle;
class vtkInteractorStyleRubberBandZoom;

/**
 * @brief Tool for rubber band zoom (zoom to box) functionality
 *
 * This tool enables interactive zoom by drawing a rubber band box
 * over a region of interest. The camera will zoom to fit the selected region.
 */
class QPCL_ENGINE_LIB_API cvZoomToBoxTool : public QObject {
    Q_OBJECT

public:
    explicit cvZoomToBoxTool(QObject* parent = nullptr);
    virtual ~cvZoomToBoxTool();

    /**
     * @brief Set the visualizer for this tool
     * @param viewer Pointer to the generic 3D visualizer
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer);

    /**
     * @brief Enable zoom to box mode
     * Switches the interactor style to rubber band zoom mode
     */
    void enable();

    /**
     * @brief Disable zoom to box mode
     * Restores the previous interactor style
     */
    void disable();

    /**
     * @brief Check if zoom to box mode is currently enabled
     */
    bool isEnabled() const { return m_enabled; }

    /**
     * @brief Get the zoom cursor
     */
    QCursor getZoomCursor() const { return m_zoomCursor; }

signals:
    /**
     * @brief Emitted when zoom to box operation is completed
     */
    void zoomCompleted();

    /**
     * @brief Emitted when the tool state changes
     */
    void enabledChanged(bool enabled);

protected:
    /**
     * @brief Store the current interactor style before switching
     */
    void storeCurrentStyle();

    /**
     * @brief Restore the previous interactor style
     */
    void restoreStyle();

private:
    ecvGenericVisualizer3D* m_viewer;
    vtkRenderWindowInteractor* m_interactor;
    vtkRenderer* m_renderer;

    vtkSmartPointer<vtkInteractorStyle> m_previousStyle;
    vtkSmartPointer<vtkInteractorStyleRubberBandZoom> m_zoomStyle;

    bool m_enabled;
    QCursor m_zoomCursor;

    unsigned long m_observerId;
};
