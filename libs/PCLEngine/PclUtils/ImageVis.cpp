// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// Local
#include "ImageVis.h"

#include <Utils/PCLConv.h>

#include "PclUtils/CustomContextItem.h"
#include "Tools/Common/PclTools.h"
#include "Tools/Common/ecvTools.h"

// CV_CORE_LIB
#include <CVPlatform.h>
#include <CVTools.h>
#include <Parallel.h>
#include <ecvGLMatrix.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Qt
#include <QImage>

// ECV_DB_LIB
#include <ecvBBox.h>

// VTK
#include <vtkAxes.h>
#include <vtkAxesActor.h>
#include <vtkBMPReader.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCaptionActor2D.h>
#include <vtkCommand.h>
#include <vtkContext2D.h>
#include <vtkInteractorObserver.h>
#include <vtkInteractorStyleImage.h>
#include <vtkJPEGReader.h>
#include <vtkLookupTable.h>
#include <vtkMath.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkPNGReader.h>
#include <vtkPNMReader.h>
#include <vtkPropAssembly.h>
#include <vtkProperty.h>
#include <vtkQImageToImageSource.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkTIFFReader.h>
#include <vtkTextProperty.h>
#include <vtkTextureUnitManager.h>
#include <vtkTransform.h>
#include <vtkUnsignedCharArray.h>

#if VTK_MAJOR_VERSION >= 6
#include <VTKExtensions/InteractionStyle/vtkPVImageInteractorStyle.h>
#include <vtkImageProperty.h>
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>
#endif

// PCL
#include <pcl/common/transforms.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/vtk/pcl_context_item.h>
#include <pcl/visualization/vtk/vtkRenderWindowInteractorFix.h>

// Support for VTK 7.1 upwards
#ifdef vtkGenericDataArray_h
#define SetTupleValue SetTypedTuple
#define InsertNextTupleValue InsertNextTypedTuple
#define GetTupleValue GetTypedTuple
#endif

// SYSTEM
#include <algorithm>
#include <cstring>
#include <map>

using namespace std;

namespace PclUtils {
ImageVis::ImageVis(const string& viewerName, bool initIterator)
    : pcl::visualization::ImageViewer(viewerName), m_cameraParamsSaved(false) {
    // Initialize camera params
    m_originalCameraParams.parallelProjection = false;
    m_originalCameraParams.focalPoint[0] =
            m_originalCameraParams.focalPoint[1] =
                    m_originalCameraParams.focalPoint[2] = 0.0;
    m_originalCameraParams.position[0] = m_originalCameraParams.position[1] =
            m_originalCameraParams.position[2] = 0.0;
    m_originalCameraParams.viewUp[0] = m_originalCameraParams.viewUp[1] =
            m_originalCameraParams.viewUp[2] = 0.0;
    m_originalCameraParams.viewAngle = 30.0;
    m_originalCameraParams.parallelScale = 1.0;
    m_originalCameraParams.clippingRange[0] = 0.01;
    m_originalCameraParams.clippingRange[1] = 1000.0;
}

vtkSmartPointer<vtkRenderWindow> ImageVis::getRenderWindow() {
    return this->win_;
}

void ImageVis::setupInteractor(
        vtkSmartPointer<vtkRenderWindowInteractor> interactor,
        vtkSmartPointer<vtkRenderWindow> win) {
    if (!win || !interactor) {
        return;
    }

    setRenderWindow(win);
    setRenderWindowInteractor(interactor);
    getRenderWindow()->Render();
}

void ImageVis::enable2Dviewer(bool state) {
#ifdef CV_LINUX
    CVLog::Warning(
            "[ImageVis::enable2Dviewer] Do not support 2D viewer on Linux or "
            "Mac platform now!");
    return;
#endif
    if (state) {
        m_mainInteractor = getRenderWindowInteractor();
        setRenderWindowInteractor(
                vtkSmartPointer<vtkRenderWindowInteractor>::Take(
                        vtkRenderWindowInteractorFixNew()));
        getRenderWindow()->SetInteractor(getRenderWindowInteractor());
        getRenderWindowInteractor()->SetRenderWindow(getRenderWindow());
        m_mouseConnection =
                registerMouseCallback(&ImageVis::mouseEventProcess, *this);
    } else {
        setupInteractor(m_mainInteractor, getRenderWindow());
        getRenderWindow()->SetInteractor(getRenderWindowInteractor());
        getRenderWindowInteractor()->SetRenderWindow(getRenderWindow());
        m_mouseConnection.disconnect();
    }
}

void ImageVis::mouseEventProcess(const pcl::visualization::MouseEvent& event,
                                 void* args) {
    if (event.getButton() == pcl::visualization::MouseEvent::RightButton &&
        event.getType() == pcl::visualization::MouseEvent::MouseMove) {
        std::string id = pickItem(event);
        if (id != "") {
            CVLog::Print(QString("Picked item id : %1").arg(id.c_str()));
        }
    }
}

std::string ImageVis::pickItem(const pcl::visualization::MouseEvent& event) {
    int x = event.getX();
    int y = event.getY();

    return pickItem(x, y);
}

std::string ImageVis::pickItem(int x, int y) {
    for (int i = 0; i < layer_map_.size(); ++i) {
        Layer* layer = &layer_map_[i];
        int index = 0;
        while (layer->actor->GetScene()->GetItem(index)) {
            pcl::visualization::context_items::Rectangle* context =
                    reinterpret_cast<
                            pcl::visualization::context_items::Rectangle*>(
                            layer->actor->GetScene()->GetItem(index));
            if (context && context->params.size() == 4) {
                bool containFlag =
                        (x >= context->params[0] &&
                         x <= context->params[0] + context->params[2] &&
                         y >= context->params[1] &&
                         y <= context->params[1] + context->params[3]);
                if (containFlag) {
                    return layer->layer_name;
                }
            }
            index++;
        }
    }
    return std::string("");
}

void ImageVis::setRenderWindow(vtkSmartPointer<vtkRenderWindow> win) {
    this->win_ = win;

    // Add window resize observer
    if (win && !m_windowResizeCallback) {
        m_windowResizeCallback = vtkSmartPointer<vtkCallbackCommand>::New();
        m_windowResizeCallback->SetCallback(WindowResizeCallback);
        m_windowResizeCallback->SetClientData(this);
        win->AddObserver(vtkCommand::WindowResizeEvent, m_windowResizeCallback);
    }
}

vtkSmartPointer<vtkRenderWindowInteractor>
ImageVis::getRenderWindowInteractor() {
    return this->interactor_;
}

void ImageVis::setRenderWindowInteractor(
        vtkSmartPointer<vtkRenderWindowInteractor> interactor) {
    this->interactor_ = interactor;

    // Save original interactor style if not already saved
    // This is needed to prevent crashes when getCamera is called
    if (interactor && !m_originalInteractorStyle) {
        m_originalInteractorStyle = interactor->GetInteractorStyle();
    }

    timer_id_ = this->interactor_->CreateRepeatingTimer(5000L);

    // Set the exit callbacks
    exit_main_loop_timer_callback_ =
            vtkSmartPointer<ExitMainLoopTimerCallback>::New();
    exit_main_loop_timer_callback_->window = this;
    exit_main_loop_timer_callback_->right_timer_id = -1;
    this->interactor_->AddObserver(vtkCommand::TimerEvent,
                                   exit_main_loop_timer_callback_);

    exit_callback_ = vtkSmartPointer<ExitCallback>::New();
    exit_callback_->window = this;
    this->interactor_->AddObserver(vtkCommand::ExitEvent, exit_callback_);

    resetStoppedFlag();
}

vtkSmartPointer<vtkRenderer> ImageVis::getRender() { return this->ren_; }

void ImageVis::setRender(vtkSmartPointer<vtkRenderer> render) {
    this->ren_ = render;
    this->ren_->AddViewProp(slice_);
}

bool ImageVis::contains(const std::string& id) const {
    // Check layer_map_ first
    LayerMap::const_iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(id));
    if (am_it != layer_map_.end()) {
        return true;
    }

    // Also check m_imageInfoMap for image layers
#if VTK_MAJOR_VERSION >= 6
    auto it = m_imageInfoMap.find(id);
    if (it != m_imageInfoMap.end() && it->second.imageSlice) {
        return true;
    }
#endif

    return false;
}

pcl::visualization::ImageViewer::Layer* ImageVis::getLayer(
        const std::string& id) {
    for (auto& l : layer_map_) {
        if (l.layer_name == id) {
            return &l;
        }
    }
    return nullptr;
}

void ImageVis::hideShowActors(bool visibility, const std::string& viewID) {
    double opacity = visibility ? 1.0 : 0.0;

    // Handle regular layers in layer_map_
    Layer* layer = getLayer(viewID);
    if (layer) {
        int index = 0;
        while (layer->actor->GetScene()->GetItem(index)) {
            layer->actor->GetScene()->GetItem(index)->SetVisible(visibility);
            index++;
        }
        layer->actor->SetVisibility(opacity);
        layer->actor->Modified();
    }

    // Handle image layers in m_imageInfoMap
#if VTK_MAJOR_VERSION >= 6
    auto it = m_imageInfoMap.find(viewID);
    if (it != m_imageInfoMap.end() && it->second.imageSlice) {
        vtkImageSlice* imageSlice = it->second.imageSlice;
        imageSlice->SetVisibility(visibility ? 1 : 0);
        imageSlice->Modified();
    }
#endif
}

void ImageVis::changeOpacity(double opacity, const std::string& viewID) {
#if VTK_MAJOR_VERSION >= 6
    // ParaView-style: Use vtkImageSlice::GetProperty()->SetOpacity()
    // This is much more efficient - no image data modification needed
    auto it = m_imageInfoMap.find(viewID);
    if (it != m_imageInfoMap.end() && it->second.imageSlice) {
        vtkImageSlice* imageSlice = it->second.imageSlice;
        imageSlice->SetVisibility(opacity > 0.0 ? 1 : 0);
        imageSlice->GetProperty()->SetOpacity(opacity);

        CVLog::PrintDebug(
                "[ImageVis::changeOpacity] Set opacity to %f using "
                "vtkImageSlice::GetProperty()->SetOpacity()",
                opacity);

        // Trigger render to update display
        if (win_) {
            win_->Render();
        }
        return;  // Image layer handled
    }
#endif

    // Fallback: Handle regular layers in layer_map_
    Layer* layer = getLayer(viewID);
    if (layer) {
        layer->actor->SetVisibility(opacity > 0.0 ? 1 : 0);
        layer->actor->Modified();
        if (win_) {
            win_->Render();
        }
    }
}

void ImageVis::removeLayer(const std::string& layer_id) {
#if VTK_MAJOR_VERSION >= 6
    // Check if this is an image layer
    bool isImageLayer = m_imageInfoMap.find(layer_id) != m_imageInfoMap.end();

    // Remove image slice from renderer if it exists
    auto it = m_imageInfoMap.find(layer_id);
    if (it != m_imageInfoMap.end() && it->second.imageSlice && ren_) {
        ren_->RemoveViewProp(it->second.imageSlice);
    }

    // Remove from image info map
    m_imageInfoMap.erase(layer_id);

    // If this was an image layer and no more image layers exist, restore
    // original interactor style This allows 3D objects to rotate normally again
    if (isImageLayer && m_imageInfoMap.empty() && interactor_ &&
        m_originalInteractorStyle) {
        interactor_->SetInteractorStyle(m_originalInteractorStyle);
        // Clear saved style so it can be saved again if needed
        m_originalInteractorStyle = nullptr;

        // Restore original camera parameters
        if (m_cameraParamsSaved && ren_ && ren_->GetActiveCamera()) {
            vtkCamera* camera = ren_->GetActiveCamera();
            camera->SetFocalPoint(m_originalCameraParams.focalPoint);
            camera->SetPosition(m_originalCameraParams.position);
            camera->SetViewUp(m_originalCameraParams.viewUp);
            camera->SetViewAngle(m_originalCameraParams.viewAngle);
            camera->SetParallelScale(m_originalCameraParams.parallelScale);
            camera->SetClippingRange(m_originalCameraParams.clippingRange);
            if (m_originalCameraParams.parallelProjection) {
                camera->ParallelProjectionOn();
            } else {
                camera->ParallelProjectionOff();
            }
            m_cameraParamsSaved = false;
        }
    }
#endif

    // Call base class implementation
    pcl::visualization::ImageViewer::removeLayer(layer_id);
}

//////////////////////////////////////////////////////////////////////////////////////////
pcl::visualization::ImageViewer::LayerMap::iterator ImageVis::createLayer(
        const std::string& layer_id,
        int x,
        int y,
        int width,
        int height,
        double opacity,
        bool fill_box) {
    Layer l;
    l.layer_name = layer_id;
    // Create a new layer
    l.actor = vtkSmartPointer<vtkContextActor>::New();
    l.actor->PickableOff();
    l.actor->DragableOff();
    if (fill_box) {
        vtkSmartPointer<pcl::visualization::context_items::FilledRectangle>
                rect = vtkSmartPointer<pcl::visualization::context_items::
                                               FilledRectangle>::New();
        rect->setColors(0, 0, 0);
        rect->setOpacity(opacity);
        rect->set(x, y, static_cast<float>(width), static_cast<float>(height));
        l.actor->GetScene()->AddItem(rect);
    }
#if VTK_MAJOR_VERSION < 6
    image_viewer_->GetRenderer()->AddActor(l.actor);
#else
    ren_->AddActor(l.actor);
#endif
    // Add another element
    layer_map_.push_back(l);

    return (layer_map_.end() - 1);
}

void ImageVis::addRGBImage(const QImage& qimage,
                           unsigned x,
                           unsigned y,
                           const std::string& layer_id,
                           double opacity) {
    if (qimage.isNull()) {
        CVLog::Warning("[ImageVis::addRGBImage] QImage is null!");
        return;
    }

#if VTK_MAJOR_VERSION >= 6
    unsigned width = qimage.width();
    unsigned height = qimage.height();

    // Convert QImage to RGBA8888 format for transparency support
    QImage rgbaImage = qimage;
    if (rgbaImage.format() != QImage::Format_RGBA8888) {
        rgbaImage = rgbaImage.convertToFormat(QImage::Format_RGBA8888);
    }

    // Use vtkQImageToImageSource for efficient conversion (ParaView-style)
    vtkSmartPointer<vtkQImageToImageSource> qimageToImageSource =
            vtkSmartPointer<vtkQImageToImageSource>::New();
    qimageToImageSource->SetQImage(&rgbaImage);
    qimageToImageSource->Update();
    vtkSmartPointer<vtkImageData> imageData = qimageToImageSource->GetOutput();

    // Ensure image data has correct origin and spacing for proper camera setup
    if (imageData) {
        imageData->SetOrigin(0.0, 0.0, 0.0);
        imageData->SetSpacing(1.0, 1.0, 1.0);
    }

    // Check if layer already exists, remove old image slice if needed
    auto it = m_imageInfoMap.find(layer_id);
    if (it != m_imageInfoMap.end() && it->second.imageSlice && ren_) {
        ren_->RemoveViewProp(it->second.imageSlice);
    }

    // Create vtkImageSliceMapper (ParaView-style)
    vtkSmartPointer<vtkImageSliceMapper> mapper =
            vtkSmartPointer<vtkImageSliceMapper>::New();
    mapper->SetInputData(imageData);
    mapper->SetSliceNumber(0);  // 2D image, use slice 0
    mapper->Update();           // Ensure mapper is updated

    // Create vtkImageSlice (ParaView-style)
    vtkSmartPointer<vtkImageSlice> imageSlice =
            vtkSmartPointer<vtkImageSlice>::New();
    imageSlice->SetMapper(mapper);
    imageSlice->GetProperty()->SetOpacity(opacity);
    imageSlice->GetProperty()->SetInterpolationTypeToLinear();

    // Set Position as requested
    imageSlice->SetPosition(x, y, 0);

    // Enable transparency rendering support
    if (opacity < 1.0 && ren_) {
        vtkRenderWindow* renderWindow = ren_->GetRenderWindow();
        if (renderWindow) {
            renderWindow->SetAlphaBitPlanes(1);
        }
    }

    // Hide base class slice_ if it exists to avoid conflict with new imageSlice
    // Base class slice_ is added in setRender(), but we use our own imageSlice
    // for each layer
    if (ren_ && slice_) {
        slice_->SetVisibility(0);  // Hide base class slice_
    }

    // Set interactor style to image style (pan/drag instead of rotate)
    // This saves original style to prevent crashes when getCamera is called
    setImageInteractorStyle();

    // Add image slice to renderer first (before calling
    // updateImageSliceTransform)
    if (ren_) {
        ren_->AddViewProp(imageSlice);
    }

    // Calculate scale and position to fit window while maintaining aspect ratio
    // This must be called after adding to renderer so ResetCamera() works
    // correctly
    updateImageSliceTransform(imageSlice, width, height);

    // Trigger render to update display
    if (win_) {
        win_->Render();
    }

    // Store image info for later updates
    ImageInfo info;
    info.originalWidth = width;
    info.originalHeight = height;
    info.imageSlice = imageSlice;
    info.imageMapper = mapper;
    m_imageInfoMap[layer_id] = info;
#else
    // Fallback for older VTK versions - use old method
    CVLog::Warning("[ImageVis::addRGBImage] vtkImageSlice requires VTK >= 6.0");
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////
// ParaView-style: Use vtkImageSlice + vtkImageSliceMapper for efficient image
// rendering
void ImageVis::addQImage(const QImage& qimage,
                         const std::string& layer_id,
                         double opacity) {
    addRGBImage(qimage, 0, 0, layer_id, opacity);
}

bool ImageVis::addText(unsigned int x,
                       unsigned int y,
                       const std::string& text_string,
                       double r,
                       double g,
                       double b,
                       const std::string& layer_id,
                       double opacity,
                       int fontSize,
                       bool bold) {
    // bool sucess = pcl::visualization::ImageViewer::addText(x, y, text_string,
    // r, g, b, layer_id, opacity);

    // Check to see if this ID entry already exists (has it been already added
    // to the visualizer?)
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        PCL_DEBUG(
                "[pcl::visualization::ImageViewer::addText] No layer with "
                "ID='%s' found. Creating new one...\n",
                layer_id.c_str());
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
#if ((VTK_MAJOR_VERSION == 5) && (VTKOR_VERSION > 10))
        interactor_style_->adjustCamera(ren_);
#endif
    }

    vtkSmartPointer<PclUtils::context_items::Text> text =
            vtkSmartPointer<PclUtils::context_items::Text>::New();
    text->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    text->setOpacity(opacity);
    text->setBold(bold);
    text->setFontSize(fontSize);
#if ((VTK_MAJOR_VERSION >= 6) || \
     ((VTK_MAJOR_VERSION == 5) && (VTK_MINOR_VERSION > 7)))
    text->set(static_cast<float>(x), static_cast<float>(y), text_string);
#else
    text->set(static_cast<float>(x), static_cast<float>(getSize()[1] - y),
              text_string);
#endif
    am_it->actor->GetScene()->AddItem(text);

    return true;
}

void ImageVis::WindowResizeCallback(vtkObject* caller,
                                    unsigned long eventId,
                                    void* clientData,
                                    void* callData) {
    Q_UNUSED(callData);
    Q_UNUSED(eventId);

    ImageVis* self = static_cast<ImageVis*>(clientData);
    if (self) {
        self->onWindowResize();
    }
}

void ImageVis::onWindowResize() {
    updateImageScales();
    if (win_) {
        win_->Render();
    }
}

void ImageVis::updateImageScales() {
    if (!win_) {
        return;
    }

    int* winSize = win_->GetSize();
    if (!winSize || winSize[0] <= 0 || winSize[1] <= 0) {
        return;
    }

#if VTK_MAJOR_VERSION >= 6
    // Update scale and position for all image slices (ParaView-style)
    for (auto& pair : m_imageInfoMap) {
        if (pair.second.imageSlice && pair.second.originalWidth > 0 &&
            pair.second.originalHeight > 0) {
            updateImageSliceTransform(pair.second.imageSlice,
                                      pair.second.originalWidth,
                                      pair.second.originalHeight);
        }
    }
#else
    // Fallback for older VTK versions
    for (auto& pair : m_imageInfoMap) {
        if (pair.second.imageItem) {
            pair.second.imageItem->updateScale(winSize[0], winSize[1]);
        }
    }
#endif
}

#if VTK_MAJOR_VERSION >= 6
void ImageVis::updateImageSliceTransform(vtkImageSlice* imageSlice,
                                         unsigned width,
                                         unsigned height) {
    if (!imageSlice || !win_ || !ren_ || width == 0 || height == 0) {
        return;
    }

    int* winSize = win_->GetSize();
    if (!winSize || winSize[0] <= 0 || winSize[1] <= 0) {
        return;
    }

    // Set display extent to show the full image
    vtkImageSliceMapper* mapper =
            vtkImageSliceMapper::SafeDownCast(imageSlice->GetMapper());
    if (mapper) {
        int extent[6] = {0, static_cast<int>(width) - 1,
                         0, static_cast<int>(height) - 1,
                         0, 0};
        mapper->SetDisplayExtent(extent);
        mapper->SetBorder(1);  // Enable border to ensure image is visible
        mapper->Update();      // Ensure mapper updates after setting extent
    }

    // Set up camera similar to base class ImageViewer::addRGBImage
    // The image slice position is at origin (0,0,0) by default
    // We use camera to view the image properly
    vtkCamera* camera = ren_->GetActiveCamera();
    if (camera) {
        // Get image slice position
        double pos[3] = {0.0, 0.0, 0.0};
        if (imageSlice) {
            imageSlice->GetPosition(pos);
        }

        // Set camera to view the image centered
        // Image extent is [0, width-1, 0, height-1, 0, 0]
        double xc = pos[0] + (width - 1) * 0.5;
        double yc = pos[1] + (height - 1) * 0.5;
        double zd = 3.346065;  // Default distance used in base class

        camera->SetFocalPoint(xc, yc, 0.0);
        camera->SetPosition(xc, yc, zd);
        // Use parallel scale based on image height (similar to base class)
        camera->SetParallelScale(0.5 * height);
        camera->ParallelProjectionOn();

        // Reset camera to ensure proper view
        ren_->ResetCamera();
    }

    // Mark image slice as modified to trigger rendering update
    if (imageSlice) {
        imageSlice->Modified();
    }
}

void ImageVis::setImageInteractorStyle() {
    // Set interactor style to image style (pan/drag instead of rotate) when
    // image is loaded vtkInteractorStyleImage supports pan (middle button) and
    // zoom (wheel) but not rotate This prevents crashes by saving original
    // style and checking before setting
    if (!interactor_) {
        return;
    }

    // Save original style if not already saved
    // This is critical to prevent crashes when PCLVis::getCamera tries to
    // access PCLVisualizerInteractorStyle methods that don't exist in
    // vtkInteractorStyleImage
    if (!m_originalInteractorStyle) {
        m_originalInteractorStyle = interactor_->GetInteractorStyle();
    }

    // Save original camera parameters if not already saved
    // This allows restoring all camera parameters when exiting image mode
    if (!m_cameraParamsSaved && ren_ && ren_->GetActiveCamera()) {
        vtkCamera* camera = ren_->GetActiveCamera();
        m_originalCameraParams.parallelProjection =
                camera->GetParallelProjection();
        camera->GetFocalPoint(m_originalCameraParams.focalPoint);
        camera->GetPosition(m_originalCameraParams.position);
        camera->GetViewUp(m_originalCameraParams.viewUp);
        m_originalCameraParams.viewAngle = camera->GetViewAngle();
        m_originalCameraParams.parallelScale = camera->GetParallelScale();
        camera->GetClippingRange(m_originalCameraParams.clippingRange);
        m_cameraParamsSaved = true;

        // Set parallel projection for image viewing
        camera->ParallelProjectionOn();
    }

    // Only set image style if current style is not already an image style
    // This avoids unnecessary style changes and potential issues
    vtkInteractorStyleImage* currentImageStyle =
            vtkInteractorStyleImage::SafeDownCast(
                    interactor_->GetInteractorStyle());
    if (!currentImageStyle) {
        // Use ParaView-style image interactor: left button pan, middle button
        // rotate
        vtkSmartPointer<vtkPVImageInteractorStyle> imageStyle =
                vtkSmartPointer<vtkPVImageInteractorStyle>::New();
        imageStyle->SetDefaultRenderer(ren_);
        interactor_->SetInteractorStyle(imageStyle);
    }
}
#endif
}  // namespace PclUtils
