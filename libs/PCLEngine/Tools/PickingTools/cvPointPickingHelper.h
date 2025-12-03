// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QKeySequence>
#include <QObject>
#include <QPointer>
#include <QShortcut>

#include <vtkSmartPointer.h>

class QWidget;
class vtkCellPicker;
class vtkPointPicker;
class vtkRenderWindowInteractor;
class vtkRenderer;

/**
 * @brief cvPointPickingHelper is a helper class for supporting keyboard
 * shortcut-based point picking in measurement tools.
 *
 * This class is inspired by ParaView's pqPointPickingHelper and provides
 * keyboard shortcuts for picking points on mesh surfaces.
 *
 * Usage:
 * - '1' / 'Ctrl+1': Pick Point 1 (surface / snap to mesh point)
 * - '2' / 'Ctrl+2': Pick Point 2 (surface / snap to mesh point)
 * - 'C' / 'Ctrl+C': Pick Center point (for angle tools, surface / snap)
 * - 'P' / 'Ctrl+P': Pick alternating points (surface / snap to mesh point)
 * - 'N': Pick point and set normal direction
 */
class cvPointPickingHelper : public QObject {
    Q_OBJECT

public:
    enum PickOption {
        Coordinates,       ///< Pick point coordinates only
        Normal,            ///< Pick normal only
        CoordinatesAndNormal  ///< Pick both coordinates and normal
    };

    /**
     * @brief Constructor
     * @param keySequence The keyboard shortcut to trigger picking
     * @param pickOnMesh If true, snap to the closest mesh point; otherwise pick on surface
     * @param parent Parent widget that receives the shortcut
     * @param pickOpt What to pick (coordinates, normal, or both)
     */
    cvPointPickingHelper(const QKeySequence& keySequence,
                         bool pickOnMesh,
                         QWidget* parent = nullptr,
                         PickOption pickOpt = Coordinates);
    ~cvPointPickingHelper() override;

    /**
     * @brief Returns whether picking snaps to mesh points
     */
    bool pickOnMesh() const { return m_pickOnMesh; }

    /**
     * @brief Returns the pick option
     */
    PickOption getPickOption() const { return m_pickOption; }

    /**
     * @brief Set the VTK interactor for picking
     */
    void setInteractor(vtkRenderWindowInteractor* interactor);

    /**
     * @brief Set the VTK renderer for picking
     */
    void setRenderer(vtkRenderer* renderer);

    /**
     * @brief Set the context widget for the shortcut
     */
    void setContextWidget(QWidget* widget);

    /**
     * @brief Enable or disable the shortcut
     * @param enabled Whether to enable the shortcut
     * @param setFocus If true and enabling, set focus to the shortcut's parent widget
     */
    void setEnabled(bool enabled, bool setFocus = false);

    /**
     * @brief Check if shortcut is enabled
     */
    bool isEnabled() const;

Q_SIGNALS:
    /**
     * @brief Emitted when a point is picked
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     */
    void pick(double x, double y, double z);

    /**
     * @brief Emitted when a point and normal are picked
     * @param px Point X coordinate
     * @param py Point Y coordinate
     * @param pz Point Z coordinate
     * @param nx Normal X component
     * @param ny Normal Y component
     * @param nz Normal Z component
     */
    void pickNormal(double px, double py, double pz,
                    double nx, double ny, double nz);

private Q_SLOTS:
    void pickPoint();

private:
    Q_DISABLE_COPY(cvPointPickingHelper)

    QPointer<QShortcut> m_shortcut;
    QPointer<QWidget> m_contextWidget;
    vtkRenderWindowInteractor* m_interactor = nullptr;
    vtkRenderer* m_renderer = nullptr;
    bool m_pickOnMesh;
    PickOption m_pickOption;
    
    // Reusable pickers to avoid creating new objects on each pick
    vtkSmartPointer<vtkPointPicker> m_pointPicker;
    vtkSmartPointer<vtkCellPicker> m_cellPicker;
};

