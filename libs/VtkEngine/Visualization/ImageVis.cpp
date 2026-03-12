// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file ImageVis.cpp
 * @brief Implementation of image visualization with VTK context.
 */

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// Local
#include "ImageVis.h"

#include <VtkRendering/Context/ContextItems.h>

#include "Tools/Common/ecvTools.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVPlatform.h>
#include <CVTools.h>

// Qt
#include <QImage>

// CV_DB_LIB
#include <ecvBBox.h>

// VTK
#include <VTKExtensions/InteractionStyle/vtkPVImageInteractorStyle.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkContext2D.h>
#include <vtkContextScene.h>
#include <vtkImageData.h>
#include <vtkImageProperty.h>
#include <vtkImageSlice.h>
#include <vtkImageSliceMapper.h>
#include <vtkInteractorObserver.h>
#include <vtkInteractorStyleImage.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkQImageToImageSource.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkTextProperty.h>

// VtkRendering interactor fix
#include <VtkRendering/Fixes/vtkRenderWindowInteractorFix.h>

// SYSTEM
#include <algorithm>
#include <cstring>
#include <map>

using namespace std;

namespace Visualization {
ImageVis::ImageVis(const string& viewerName, bool /*autoInit*/)
    : VtkRendering::ImageVisualizer(viewerName), m_cameraParamsSaved(false) {
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

void ImageVis::mouseEventProcess(const VtkRendering::MouseEvent& event,
                                 void* /*args*/) {
    if (event.getButton() == VtkRendering::MouseEvent::RightButton &&
        event.getType() == VtkRendering::MouseEvent::MouseMove) {
        std::string id = pickItem(event);
        if (!id.empty()) {
            CVLog::Print(QString("Picked item id : %1").arg(id.c_str()));
        }
    }
}

std::string ImageVis::pickItem(const VtkRendering::MouseEvent& event) {
    return pickItem(event.getX(), event.getY());
}

std::string ImageVis::pickItem(int x, int y) {
    for (size_t i = 0; i < layer_map_.size(); ++i) {
        Layer* layer = &layer_map_[i];
        int index = 0;
        while (layer->actor->GetScene()->GetItem(index)) {
            auto* context =
                    reinterpret_cast<VtkRendering::context_items::Rectangle*>(
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

    if (interactor && !m_originalInteractorStyle) {
        m_originalInteractorStyle = interactor->GetInteractorStyle();
    }

    timer_id_ = this->interactor_->CreateRepeatingTimer(5000L);

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
    LayerMap::const_iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(id));
    if (am_it != layer_map_.end()) {
        return true;
    }

    auto it = m_imageInfoMap.find(id);
    if (it != m_imageInfoMap.end() && it->second.imageSlice) {
        return true;
    }

    return false;
}

VtkRendering::ImageVisualizer::Layer* ImageVis::getLayer(
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

    auto it = m_imageInfoMap.find(viewID);
    if (it != m_imageInfoMap.end() && it->second.imageSlice) {
        vtkImageSlice* imageSlice = it->second.imageSlice;
        imageSlice->SetVisibility(visibility ? 1 : 0);
        imageSlice->Modified();
    }
}

void ImageVis::changeOpacity(double opacity, const std::string& viewID) {
    auto it = m_imageInfoMap.find(viewID);
    if (it != m_imageInfoMap.end() && it->second.imageSlice) {
        vtkImageSlice* imageSlice = it->second.imageSlice;
        imageSlice->SetVisibility(opacity > 0.0 ? 1 : 0);
        imageSlice->GetProperty()->SetOpacity(opacity);

        CVLog::PrintVerbose(
                "[ImageVis::changeOpacity] Set opacity to %f using "
                "vtkImageSlice::GetProperty()->SetOpacity()",
                opacity);

        if (win_) {
            win_->Render();
        }
        return;
    }

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
    bool isImageLayer = m_imageInfoMap.find(layer_id) != m_imageInfoMap.end();

    auto it = m_imageInfoMap.find(layer_id);
    if (it != m_imageInfoMap.end() && it->second.imageSlice && ren_) {
        ren_->RemoveViewProp(it->second.imageSlice);
    }

    m_imageInfoMap.erase(layer_id);

    if (isImageLayer && m_imageInfoMap.empty() && interactor_ &&
        m_originalInteractorStyle) {
        interactor_->SetInteractorStyle(m_originalInteractorStyle);
        m_originalInteractorStyle = nullptr;

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

    VtkRendering::ImageVisualizer::removeLayer(layer_id);
}

VtkRendering::ImageVisualizer::LayerMap::iterator ImageVis::createLayer(
        const std::string& layer_id,
        int x,
        int y,
        int width,
        int height,
        double opacity,
        bool fill_box) {
    Layer l;
    l.layer_name = layer_id;
    l.actor = vtkSmartPointer<vtkContextActor>::New();
    l.actor->PickableOff();
    l.actor->DragableOff();
    if (fill_box) {
        auto rect = vtkSmartPointer<
                VtkRendering::context_items::FilledRectangle>::New();
        rect->setColors(0, 0, 0);
        rect->setOpacity(opacity);
        rect->set(x, y, static_cast<float>(width), static_cast<float>(height));
        l.actor->GetScene()->AddItem(rect);
    }
    ren_->AddActor(l.actor);
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

    unsigned width = qimage.width();
    unsigned height = qimage.height();

    QImage rgbaImage = qimage;
    if (rgbaImage.format() != QImage::Format_RGBA8888) {
        rgbaImage = rgbaImage.convertToFormat(QImage::Format_RGBA8888);
    }

    vtkSmartPointer<vtkQImageToImageSource> qimageToImageSource =
            vtkSmartPointer<vtkQImageToImageSource>::New();
    qimageToImageSource->SetQImage(&rgbaImage);
    qimageToImageSource->Update();
    vtkSmartPointer<vtkImageData> imageData = qimageToImageSource->GetOutput();

    if (imageData) {
        imageData->SetOrigin(0.0, 0.0, 0.0);
        imageData->SetSpacing(1.0, 1.0, 1.0);
    }

    auto it = m_imageInfoMap.find(layer_id);
    if (it != m_imageInfoMap.end() && it->second.imageSlice && ren_) {
        ren_->RemoveViewProp(it->second.imageSlice);
    }

    vtkSmartPointer<vtkImageSliceMapper> mapper =
            vtkSmartPointer<vtkImageSliceMapper>::New();
    mapper->SetInputData(imageData);
    mapper->SetSliceNumber(0);
    mapper->Update();

    vtkSmartPointer<vtkImageSlice> imageSlice =
            vtkSmartPointer<vtkImageSlice>::New();
    imageSlice->SetMapper(mapper);
    imageSlice->GetProperty()->SetOpacity(opacity);
    imageSlice->GetProperty()->SetInterpolationTypeToLinear();
    imageSlice->SetPosition(x, y, 0);

    if (opacity < 1.0 && ren_) {
        vtkRenderWindow* renderWindow = ren_->GetRenderWindow();
        if (renderWindow) {
            renderWindow->SetAlphaBitPlanes(1);
        }
    }

    if (ren_ && slice_) {
        slice_->SetVisibility(0);
    }

    setImageInteractorStyle();

    if (ren_) {
        ren_->AddViewProp(imageSlice);
    }

    updateImageSliceTransform(imageSlice, width, height);

    if (win_) {
        win_->Render();
    }

    ImageInfo info;
    info.originalWidth = width;
    info.originalHeight = height;
    info.imageSlice = imageSlice;
    info.imageMapper = mapper;
    m_imageInfoMap[layer_id] = info;
}

void ImageVis::addQImage(const QImage& qimage,
                         const std::string& layer_id,
                         double opacity) {
    addRGBImage(qimage, 0, 0, layer_id, opacity);
}

bool ImageVis::addLine(unsigned int x_min,
                       unsigned int y_min,
                       unsigned int x_max,
                       unsigned int y_max,
                       double r,
                       double g,
                       double b,
                       const std::string& layer_id,
                       double opacity) {
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
    }

    auto line = vtkSmartPointer<VtkRendering::context_items::Line>::New();
    line->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    line->setOpacity(opacity);
    line->set(static_cast<float>(x_min), static_cast<float>(y_min),
              static_cast<float>(x_max), static_cast<float>(y_max));
    am_it->actor->GetScene()->AddItem(line);
    return true;
}

bool ImageVis::addRectangle(unsigned int x_min,
                            unsigned int x_max,
                            unsigned int y_min,
                            unsigned int y_max,
                            double r,
                            double g,
                            double b,
                            const std::string& layer_id,
                            double opacity) {
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
    }

    auto rect = vtkSmartPointer<VtkRendering::context_items::Rectangle>::New();
    rect->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    rect->setOpacity(opacity);
    rect->set(static_cast<float>(x_min), static_cast<float>(y_min),
              static_cast<float>(x_max), static_cast<float>(y_max));
    am_it->actor->GetScene()->AddItem(rect);
    return true;
}

bool ImageVis::addFilledRectangle(unsigned int x_min,
                                  unsigned int x_max,
                                  unsigned int y_min,
                                  unsigned int y_max,
                                  double r,
                                  double g,
                                  double b,
                                  const std::string& layer_id,
                                  double opacity) {
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
    }

    auto rect = vtkSmartPointer<
            VtkRendering::context_items::FilledRectangle>::New();
    rect->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    rect->setOpacity(opacity);
    rect->set(static_cast<float>(x_min), static_cast<float>(y_min),
              static_cast<float>(x_max - x_min),
              static_cast<float>(y_max - y_min));
    am_it->actor->GetScene()->AddItem(rect);
    return true;
}

bool ImageVis::addCircle(unsigned int x,
                         unsigned int y,
                         double radius,
                         double r,
                         double g,
                         double b,
                         const std::string& layer_id,
                         double opacity) {
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
    }

    auto circle = vtkSmartPointer<VtkRendering::context_items::Circle>::New();
    circle->setColors(static_cast<unsigned char>(255.0 * r),
                      static_cast<unsigned char>(255.0 * g),
                      static_cast<unsigned char>(255.0 * b));
    circle->setOpacity(opacity);
    circle->set(static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(radius));
    am_it->actor->GetScene()->AddItem(circle);
    return true;
}

bool ImageVis::markPoint(unsigned int x,
                         unsigned int y,
                         const Eigen::Array<unsigned char, 3, 1>& fg_color,
                         const Eigen::Array<unsigned char, 3, 1>& bg_color,
                         double radius,
                         const std::string& layer_id,
                         double opacity) {
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
    }

    auto marker = vtkSmartPointer<VtkRendering::context_items::Markers>::New();
    marker->setColors(fg_color[0], fg_color[1], fg_color[2]);
    marker->setPointColors(bg_color[0], bg_color[1], bg_color[2]);
    marker->setOpacity(opacity);
    marker->setSize(static_cast<float>(radius));
    std::vector<float> xy = {static_cast<float>(x), static_cast<float>(y)};
    marker->set(xy);
    am_it->actor->GetScene()->AddItem(marker);
    return true;
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
    LayerMap::iterator am_it = std::find_if(
            layer_map_.begin(), layer_map_.end(), LayerComparator(layer_id));
    if (am_it == layer_map_.end()) {
        am_it = createLayer(layer_id, getSize()[0] - 1, getSize()[1] - 1,
                            opacity, false);
    }

    auto text = vtkSmartPointer<VtkRendering::context_items::Text>::New();
    text->setColors(static_cast<unsigned char>(255.0 * r),
                    static_cast<unsigned char>(255.0 * g),
                    static_cast<unsigned char>(255.0 * b));
    text->setOpacity(opacity);
    text->setBold(bold);
    text->setFontSize(fontSize);
    text->set(static_cast<float>(x), static_cast<float>(y), text_string);
    am_it->actor->GetScene()->AddItem(text);

    return true;
}

void ImageVis::WindowResizeCallback(vtkObject* /*caller*/,
                                    unsigned long /*eventId*/,
                                    void* clientData,
                                    void* /*callData*/) {
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

    for (auto& pair : m_imageInfoMap) {
        if (pair.second.imageSlice && pair.second.originalWidth > 0 &&
            pair.second.originalHeight > 0) {
            updateImageSliceTransform(pair.second.imageSlice,
                                      pair.second.originalWidth,
                                      pair.second.originalHeight);
        }
    }
}

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

    vtkImageSliceMapper* mapper =
            vtkImageSliceMapper::SafeDownCast(imageSlice->GetMapper());
    if (mapper) {
        int extent[6] = {0, static_cast<int>(width) - 1,
                         0, static_cast<int>(height) - 1,
                         0, 0};
        mapper->SetDisplayExtent(extent);
        mapper->SetBorder(1);
        mapper->Update();
    }

    vtkCamera* camera = ren_->GetActiveCamera();
    if (camera) {
        double pos[3] = {0.0, 0.0, 0.0};
        if (imageSlice) {
            imageSlice->GetPosition(pos);
        }

        double xc = pos[0] + (width - 1) * 0.5;
        double yc = pos[1] + (height - 1) * 0.5;
        double zd = 3.346065;

        camera->SetViewUp(0.0, 1.0, 0.0);
        camera->SetFocalPoint(xc, yc, 0.0);
        camera->SetPosition(xc, yc, zd);
        camera->SetParallelScale(0.5 * height);
        camera->ParallelProjectionOn();

        ren_->ResetCameraClippingRange();
    }

    if (imageSlice) {
        imageSlice->Modified();
    }
}

void ImageVis::setImageInteractorStyle() {
    if (!interactor_) {
        return;
    }

    if (!m_originalInteractorStyle) {
        m_originalInteractorStyle = interactor_->GetInteractorStyle();
    }

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

        camera->ParallelProjectionOn();
    }

    vtkInteractorStyleImage* currentImageStyle =
            vtkInteractorStyleImage::SafeDownCast(
                    interactor_->GetInteractorStyle());
    if (!currentImageStyle) {
        vtkSmartPointer<vtkPVImageInteractorStyle> imageStyle =
                vtkSmartPointer<vtkPVImageInteractorStyle>::New();
        imageStyle->SetDefaultRenderer(ren_);
        interactor_->SetInteractorStyle(imageStyle);
    }
}
}  // namespace Visualization
