// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "PCLCloud.h"
#include "PclUtils/CustomContextItem.h"
#include "qPCL.h"

// ECV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDrawContext.h>
#include <ecvGenericVisualizer2D.h>

// PCL
#include <visualization/include/pcl/visualization/image_viewer.h>
// #include <pcl/visualization/image_viewer.h>

// VTK
#include <vtkSmartPointer.h>

// Forward declarations
class vtkRenderer;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
#if VTK_MAJOR_VERSION >= 6
class vtkImageSlice;
class vtkImageSliceMapper;
#endif

namespace PclUtils {
class QPCL_ENGINE_LIB_API ImageVis : public ecvGenericVisualizer2D,
                                     public pcl::visualization::ImageViewer {
    // Q_OBJECT
public:
    //! Default constructor
    /** Constructor is protected to avoid using this object as a non static
     *class.
     **/
    ImageVis(const std::string& viewerName, bool autoInit = false);
    /** \brief Set up our unique PCL interactor style for a given
     * vtkRenderWindowInteractor object attached to a given vtkRenderWindow
     * \param[in,out] interactor the vtkRenderWindowInteractor object to set up
     * \param[in,out] win a vtkRenderWindow object that the interactor is
     * attached to
     */
    void setupInteractor(vtkSmartPointer<vtkRenderWindowInteractor> interactor,
                         vtkSmartPointer<vtkRenderWindow> win);
    void setRenderWindow(vtkSmartPointer<vtkRenderWindow> win);
    vtkSmartPointer<vtkRenderWindow> getRenderWindow();
    vtkSmartPointer<vtkRenderWindowInteractor> getRenderWindowInteractor();
    void setRenderWindowInteractor(
            vtkSmartPointer<vtkRenderWindowInteractor> interactor);
    /** \brief The renderer. */
    vtkSmartPointer<vtkRenderer> getRender();
    void setRender(vtkSmartPointer<vtkRenderer> render);

    /** \brief Check if the image with the given id was already added to this
     * visualizer. \param[in] id the id of the image to check \return true if a
     * image with the specified id was found
     */
    bool contains(const std::string& id) const;

    Layer* getLayer(const std::string& id);
    void changeOpacity(double opacity, const std::string& viewID);
    void hideShowActors(bool visibility, const std::string& viewID);

    /** \brief Remove a layer from the viewer.
     * \param[in] layer_id the name of the layer to remove
     */
    void removeLayer(const std::string& layer_id);

    /** \brief Add a new 2D rendering layer to the viewer.
     * \param[in] layer_id the name of the layer
     * \param[in] width the width of the layer
     * \param[in] height the height of the layer
     * \param[in] opacity the opacity of the layer: 0 for invisible, 1 for
     * opaque. (default: 0.5) \param[in] fill_box set to true to fill in the
     * image with one black box before starting
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

    /** \brief Add a QImage directly (ParaView-style, using
     * vtkQImageToImageSource). \param[in] qimage the QImage to add \param[in]
     * layer_id the name of the layer \param[in] opacity the opacity of the
     * layer (default: 1.0)
     */
    void addQImage(const QImage& qimage,
                   const std::string& layer_id = "image",
                   double opacity = 1.0);

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
    void enable2Dviewer(bool state);

    std::string pickItem(int x, int y);

private:
    void mouseEventProcess(const pcl::visualization::MouseEvent& event,
                           void* args);
    boost::signals2::connection m_mouseConnection;
    std::string pickItem(const pcl::visualization::MouseEvent& event);

    // Window resize callback
    static void WindowResizeCallback(vtkObject* caller,
                                     unsigned long eventId,
                                     void* clientData,
                                     void* callData);
    void onWindowResize();
    void updateImageScales();
#if VTK_MAJOR_VERSION >= 6
    void updateImageSliceTransform(vtkImageSlice* imageSlice,
                                   unsigned width,
                                   unsigned height);
#endif

    // Set interactor style to image style (pan/drag instead of rotate)
    // This prevents crashes by saving original style and checking before
    // setting
    void setImageInteractorStyle();

    vtkSmartPointer<vtkRenderWindowInteractor> m_mainInteractor;
    vtkSmartPointer<vtkCallbackCommand> m_windowResizeCallback;

    // Store original interactor style to restore when needed
    // This prevents crashes when getCamera is called on
    // PCLVisualizerInteractorStyle
    vtkSmartPointer<vtkInteractorObserver> m_originalInteractorStyle;

    // Store original camera parameters to restore when exiting image mode
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

    // Store image slice info for each layer (ParaView-style)
    struct ImageInfo {
        unsigned originalWidth;
        unsigned originalHeight;
#if VTK_MAJOR_VERSION >= 6
        vtkSmartPointer<vtkImageSlice> imageSlice;
        vtkSmartPointer<vtkImageSliceMapper> imageMapper;
#endif
    };
    std::map<std::string, ImageInfo> m_imageInfoMap;
};

typedef std::shared_ptr<ImageVis> ImageVisPtr;
}  // namespace PclUtils
