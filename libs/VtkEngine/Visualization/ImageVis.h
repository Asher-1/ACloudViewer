// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file ImageVis.h
/// @brief 2D image visualizer for layered image display and overlay drawing.

// VtkRendering
#include <VtkRendering/Context/ImageVisualizer.h>
#include <VtkRendering/Core/MouseEvent.h>

#include "qVTK.h"

// CV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDrawContext.h>
#include <ecvGenericVisualizer2D.h>

// Eigen
#include <Eigen/Core>

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkImageSlice;
class vtkImageSliceMapper;

namespace Visualization {
/// @class ImageVis
/// @brief 2D image visualizer for layered image display, overlay drawing, and
/// picking.
class QVTK_ENGINE_LIB_API ImageVis : public ecvGenericVisualizer2D,
                                     public VtkRendering::ImageVisualizer {
public:
    /** @param viewerName Name for the viewer
     *  @param autoInit Whether to auto-initialize
     */
    ImageVis(const std::string& viewerName, bool autoInit = false);

    /** @param interactor VTK render window interactor
     *  @param win VTK render window
     */
    void setupInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor,
                         vtkSmartPointer<vtkRenderWindow> win);
    /** @param win VTK render window
     */
    void setRenderWindow(vtkSmartPointer<vtkRenderWindow> win);
    /** @return VTK render window
     */
    vtkSmartPointer<vtkRenderWindow> getRenderWindow();
    /** @return VTK render window interactor
     */
    vtkSmartPointer<vtkRenderWindowInteractor> getRenderWindowInteractor();
    /** @param interactor VTK render window interactor
     */
    void setRenderWindowInteractor(
            vtkSmartPointer<vtkRenderWindowInteractor> interactor);
    /** @return VTK renderer
     */
    vtkSmartPointer<vtkRenderer> getRender();
    /** @param render VTK renderer
     */
    void setRender(vtkSmartPointer<vtkRenderer> render);

    /** @param id Layer or item ID
     *  @return true if layer/item exists
     */
    bool contains(const std::string& id) const;

    /** @param id Layer ID
     *  @return Layer pointer or nullptr
     */
    Layer* getLayer(const std::string& id);
    /** @param opacity Opacity [0.0, 1.0]
     *  @param viewID Layer/view ID
     */
    void changeOpacity(double opacity, const std::string& viewID);
    /** @param visibility true to show, false to hide
     *  @param viewID Layer/view ID
     */
    void hideShowActors(bool visibility, const std::string& viewID);

    /** @param layer_id Layer ID to remove
     */
    void removeLayer(const std::string& layer_id);

    /** @param layer_id Unique layer identifier
     *  @param x,y,width,height Layer bounds in pixels
     *  @param opacity Layer opacity (default 0.5)
     *  @param fill_box Whether to fill the layer box
     *  @return Iterator to created layer
     */
    LayerMap::iterator createLayer(const std::string& layer_id,
                                   int x,
                                   int y,
                                   int width,
                                   int height,
                                   double opacity = 0.5,
                                   bool fill_box = true);

    void addRGBImage(const QImage& qimage,
                     unsigned x,
                     unsigned y,
                     const std::string& layer_id,
                     double opacity = 1.0);

    void addQImage(const QImage& qimage,
                   const std::string& layer_id = "image",
                   double opacity = 1.0);

    bool addLine(unsigned int x_min,
                 unsigned int y_min,
                 unsigned int x_max,
                 unsigned int y_max,
                 double r,
                 double g,
                 double b,
                 const std::string& layer_id = "line",
                 double opacity = 1.0);

    bool addRectangle(unsigned int x_min,
                      unsigned int x_max,
                      unsigned int y_min,
                      unsigned int y_max,
                      double r,
                      double g,
                      double b,
                      const std::string& layer_id = "rectangles",
                      double opacity = 1.0);

    bool addFilledRectangle(unsigned int x_min,
                            unsigned int x_max,
                            unsigned int y_min,
                            unsigned int y_max,
                            double r,
                            double g,
                            double b,
                            const std::string& layer_id = "boxes",
                            double opacity = 0.5);

    /** @param x,y Center coordinates
     *  @param radius Circle radius
     *  @param r,g,b RGB color
     *  @param layer_id Target layer (default "circles")
     *  @param opacity Opacity (default 1.0)
     *  @return true on success
     */
    bool addCircle(unsigned int x,
                   unsigned int y,
                   double radius,
                   double r,
                   double g,
                   double b,
                   const std::string& layer_id = "circles",
                   double opacity = 1.0);

    /** @param x,y Point position
     *  @param fg_color Foreground color
     *  @param bg_color Background color (default white)
     *  @param radius Mark radius (default 3.0)
     *  @param layer_id Target layer (default "points")
     *  @param opacity Opacity (default 1.0)
     *  @return true on success
     */
    bool markPoint(unsigned int x,
                   unsigned int y,
                   const Eigen::Array<unsigned char, 3, 1>& fg_color,
                   const Eigen::Array<unsigned char, 3, 1>& bg_color =
                           Eigen::Array<unsigned char, 3, 1>(255, 255, 255),
                   double radius = 3.0,
                   const std::string& layer_id = "points",
                   double opacity = 1.0);

    /** @param x,y Text position
     *  @param text_string Text content
     *  @param r,g,b RGB color
     *  @param layer_id Target layer (default "line")
     *  @param opacity Opacity (default 1.0)
     *  @param fontSize Font size (default 10)
     *  @param bold Bold text (default false)
     *  @return true on success
     */
    bool addText(unsigned int x,
                 unsigned int y,
                 const std::string& text_string,
                 double r,
                 double g,
                 double b,
                 const std::string& layer_id = "line",
                 double opacity = 1.0,
                 int fontSize = 10,
                 bool bold = false);

public:
    /** @param state true to enable 2D viewer mode
     */
    void enable2Dviewer(bool state);

    /** @param x,y Screen coordinates
     *  @return Picked layer/item ID or empty string
     */
    std::string pickItem(int x, int y);

private:
    void mouseEventProcess(const VtkRendering::MouseEvent& event, void* args);
    VtkRendering::Connection m_mouseConnection;
    std::string pickItem(const VtkRendering::MouseEvent& event);

    static void WindowResizeCallback(vtkObject* caller,
                                     unsigned long eventId,
                                     void* clientData,
                                     void* callData);
    void onWindowResize();
    void updateImageScales();
    void updateImageSliceTransform(vtkImageSlice* imageSlice,
                                   unsigned width,
                                   unsigned height);
    void setImageInteractorStyle();

    vtkSmartPointer<vtkRenderWindowInteractor> m_mainInteractor;
    vtkSmartPointer<vtkCallbackCommand> m_windowResizeCallback;
    vtkSmartPointer<vtkInteractorObserver> m_originalInteractorStyle;

    struct CameraParams {
        bool parallelProjection;
        double focalPoint[3];
        double position[3];
        double viewUp[3];
        double viewAngle;
        double parallelScale;
        double clippingRange[2];
    };
    CameraParams m_originalCameraParams;
    bool m_cameraParamsSaved;

    struct ImageInfo {
        unsigned originalWidth;
        unsigned originalHeight;
        vtkSmartPointer<vtkImageSlice> imageSlice;
        vtkSmartPointer<vtkImageSliceMapper> imageMapper;
    };
    std::map<std::string, ImageInfo> m_imageInfoMap;
};

typedef std::shared_ptr<ImageVis> ImageVisPtr;
}  // namespace Visualization
