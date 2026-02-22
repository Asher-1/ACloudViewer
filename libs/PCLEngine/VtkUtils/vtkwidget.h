// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file vtkwidget.h
 * @brief VTK visualization widget for CloudViewer
 *
 * Provides a Qt widget wrapper for VTK rendering with features:
 * - OpenGL-based 3D rendering
 * - Multi-viewport support
 * - Actor management
 * - Background color control
 * - Orientation marker
 * - Dataset visualization
 */

#pragma once

#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>

#include "qPCL.h"

class vtkActor;
class vtkProp;
class vtkLODActor;
class vtkDataSet;

/**
 * @namespace VtkUtils
 * @brief VTK utility classes and functions
 */
namespace VtkUtils {

class VtkWidgetPrivate;

/**
 * @class VtkWidget
 * @brief VTK OpenGL rendering widget
 *
 * Qt widget that provides VTK visualization capabilities with OpenGL rendering.
 * Extends QVTKOpenGLNativeWidget with additional features:
 *
 * - Multi-viewport rendering
 * - Simplified actor management
 * - Background color control
 * - Orientation marker display
 * - VTK dataset visualization
 * - Actor visibility control
 *
 * This widget provides a high-level interface to VTK rendering while
 * maintaining full access to underlying VTK components.
 *
 * @see QVTKOpenGLNativeWidget
 */
class VtkWidget : public QVTKOpenGLNativeWidget {
    Q_OBJECT
public:
    /**
     * @brief Constructor
     * @param parent Parent Qt widget (optional)
     */
    explicit VtkWidget(QWidget* parent = nullptr);

    /**
     * @brief Destructor
     *
     * Cleans up VTK resources and renderers.
     */
    virtual ~VtkWidget();

    /**
     * @brief Enable or disable multi-viewport mode
     * @param multi Enable multiple viewports (default: true)
     *
     * Multi-viewport mode allows splitting the widget into multiple renderers.
     */
    void setMultiViewports(bool multi = true);

    /**
     * @brief Check if multi-viewport mode is enabled
     * @return true if using multiple viewports
     */
    bool multiViewports() const;

    /**
     * @brief Create actor from VTK dataset
     * @param data VTK dataset to visualize
     * @param actor Output LOD actor (will be created)
     * @param use_scalars Use scalar data for coloring (default: true)
     *
     * Creates a Level-of-Detail (LOD) actor from VTK dataset with automatic
     * mapper configuration.
     */
    void createActorFromVTKDataSet(const vtkSmartPointer<vtkDataSet>& data,
                                   vtkSmartPointer<vtkLODActor>& actor,
                                   bool use_scalars = true);

    /**
     * @brief Add actor to the renderer
     * @param actor VTK actor/prop to add
     * @param clr Actor color (default: black)
     */
    void addActor(vtkProp* actor, const QColor& clr = Qt::black);

    /**
     * @brief Add view prop to renderer
     * @param prop VTK view prop to add
     *
     * More generic than addActor, accepts any vtkProp.
     */
    void addViewProp(vtkProp* prop);

    /**
     * @brief Get list of all actors
     * @return List of all VTK props in the renderer
     */
    QList<vtkProp*> actors() const;

    /**
     * @brief Set visibility for all actors
     * @param visible Visibility state for all actors
     */
    void setActorsVisible(bool visible);

    /**
     * @brief Set visibility for specific actor
     * @param actor Actor to modify
     * @param visible New visibility state
     */
    void setActorVisible(vtkProp* actor, bool visible);

    /**
     * @brief Check actor visibility
     * @param actor Actor to check
     * @return true if actor is visible
     */
    bool actorVisible(vtkProp* actor);

    /**
     * @brief Set background color
     * @param clr New background color
     */
    void setBackgroundColor(const QColor& clr);

    /**
     * @brief Reset background color to default
     */
    void setBackgroundColor();

    /**
     * @brief Get current background color
     * @return Current background color
     */
    QColor backgroundColor() const;

    /**
     * @brief Get default renderer
     * @return Pointer to default VTK renderer
     */
    vtkRenderer* defaultRenderer();

    /**
     * @brief Check if default renderer is in use
     * @return true if default renderer is taken
     */
    bool defaultRendererTaken() const;

    /**
     * @brief Show or hide orientation marker
     * @param show Show marker (default: true)
     *
     * Displays axes marker showing X/Y/Z orientation in corner of view.
     */
    void showOrientationMarker(bool show = true);

    /**
     * @brief Get VTK render window
     * @return Pointer to VTK render window
     */
    vtkRenderWindow* GetRenderWindow() { return this->renderWindow(); }

    /**
     * @brief Get VTK interactor
     * @return Pointer to VTK render window interactor
     */
    QVTKInteractor* GetInteractor() { return this->interactor(); }

protected:
    /**
     * @brief Set rendering bounds
     * @param bounds Bounds array [xmin, xmax, ymin, ymax, zmin, zmax]
     */
    void setBounds(double* bounds);

    /**
     * @brief Get minimum X bound
     * @return X minimum
     */
    double xMin() const;

    /**
     * @brief Get maximum X bound
     * @return X maximum
     */
    double xMax() const;

    /**
     * @brief Get minimum Y bound
     * @return Y minimum
     */
    double yMin() const;

    /**
     * @brief Get maximum Y bound
     * @return Y maximum
     */
    double yMax() const;

    /**
     * @brief Get minimum Z bound
     * @return Z minimum
     */
    double zMin() const;

    /**
     * @brief Get maximum Z bound
     * @return Z maximum
     */
    double zMax() const;

private:
    VtkWidgetPrivate* d_ptr;  ///< Private implementation pointer
    Q_DISABLE_COPY(VtkWidget)
};

}  // namespace VtkUtils
