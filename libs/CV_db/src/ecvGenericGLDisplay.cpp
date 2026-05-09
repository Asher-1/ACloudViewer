// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericGLDisplay.h"

#include <ecvDisplayTypes.h>
#include <ecvGuiParameters.h>

#include <QApplication>
#include <QJsonObject>
#include <QMap>
#include <QMutex>
#include <QMutexLocker>
#include <QWidget>

#include "ecvViewManager.h"

namespace {
QMutex s_registryMutex;
QMap<QWidget*, ecvGenericGLDisplay*> s_displayRegistry;
}  // namespace

// ================================================================
// Default implementations for the new per-view virtual methods.
// Concrete subclasses (vtkGLView) override these
// with real per-view logic.
// ================================================================

void ecvGenericGLDisplay::getGLCameraParameters(
        ccGLCameraParameters& /*params*/) const {}

void ecvGenericGLDisplay::getVisibleObjectsBB(ccBBox& /*box*/) const {}

void ecvGenericGLDisplay::updateConstellationCenterAndZoom(
        const ccBBox* /*box*/) {}

QRect ecvGenericGLDisplay::getGLViewport() const {
    const QWidget* w = asWidget();
    if (w) return QRect(0, 0, w->width(), w->height());
    return {};
}

int ecvGenericGLDisplay::glWidth() const { return getGLViewport().width(); }

int ecvGenericGLDisplay::glHeight() const { return getGLViewport().height(); }

int ecvGenericGLDisplay::getDevicePixelRatio() const {
    const QWidget* w = asWidget();
    return w ? static_cast<int>(w->devicePixelRatio()) : 1;
}

QJsonObject ecvGenericGLDisplay::saveLayoutCameraState() const { return {}; }

void ecvGenericGLDisplay::loadLayoutCameraState(
        const QJsonObject& /*cameraJson*/) {}

void ecvGenericGLDisplay::setInteractionMode(INTERACTION_FLAGS /*flags*/) {}

ecvGenericGLDisplay::INTERACTION_FLAGS ecvGenericGLDisplay::getInteractionMode()
        const {
    return INTERACT_NONE;
}

void ecvGenericGLDisplay::setPickingMode(PICKING_MODE /*mode*/) {}

ecvGenericGLDisplay::PICKING_MODE ecvGenericGLDisplay::getPickingMode() const {
    return NO_PICKING;
}

void ecvGenericGLDisplay::getContext(ccGLDrawContext& /*context*/) const {}

const ecvGui::ParamStruct& ecvGenericGLDisplay::getDisplayParameters() const {
    return ecvGui::Parameters();
}

void ecvGenericGLDisplay::setDisplayParameters(
        const ecvGui::ParamStruct& /*params*/, bool /*thisWindowOnly*/) {}

void ecvGenericGLDisplay::drawClickableItems(int /*xStart*/, int& /*yStart*/) {}

QFont ecvGenericGLDisplay::textDisplayFont() const {
    return QApplication::font();
}

void ecvGenericGLDisplay::display2DText(const QString& text,
                                        int x,
                                        int y,
                                        unsigned char align,
                                        float bkgAlpha,
                                        const unsigned char* rgbColor,
                                        const QFont* font,
                                        const QString& id) {
    ecvViewManager::instance().sharedDisplayText(text, x, y, align, bkgAlpha,
                                                 rgbColor, font, id, this);
}

void ecvGenericGLDisplay::moveCamera(float dx, float dy, float dz) {
    ecvViewManager::ScopedRenderOverride guard(this);
    ecvViewManager::instance().sharedMoveCamera(dx, dy, dz);
}

void ecvGenericGLDisplay::rotateBaseViewMat(const ccGLMatrixd& rotMat) {
    ecvViewManager::ScopedRenderOverride guard(this);
    ecvViewManager::instance().sharedRotateBaseViewMat(rotMat);
}

void ecvGenericGLDisplay::loadCameraParameters(const std::string& file) {
    ecvViewManager::ScopedRenderOverride guard(this);
    ecvViewManager::instance().sharedLoadCameraParameters(file);
}

void ecvGenericGLDisplay::saveCameraParameters(const std::string& file) {
    ecvViewManager::ScopedRenderOverride guard(this);
    ecvViewManager::instance().sharedSaveCameraParameters(file);
}

std::list<ccInteractor*>& ecvGenericGLDisplay::activeItemsRef() {
    static std::list<ccInteractor*> s_fallback;
    return s_fallback;
}

ecvHotZone*& ecvGenericGLDisplay::hotZonePtrRef() {
    static ecvHotZone* s_nullHz = nullptr;
    return s_nullHz;
}

std::vector<ecvClickableItem>& ecvGenericGLDisplay::clickableItemsRef() {
    static std::vector<ecvClickableItem> s_emptyClickable;
    return s_emptyClickable;
}

// ================================================================
// Static registry
// ================================================================

ecvGenericGLDisplay* ecvGenericGLDisplay::FromWidget(QWidget* widget) {
    QMutexLocker lock(&s_registryMutex);
    auto it = s_displayRegistry.find(widget);
    return (it != s_displayRegistry.end()) ? it.value() : nullptr;
}

void ecvGenericGLDisplay::RegisterGLDisplay(QWidget* widget,
                                            ecvGenericGLDisplay* display) {
    QMutexLocker lock(&s_registryMutex);
    s_displayRegistry.insert(widget, display);
}

void ecvGenericGLDisplay::UnregisterGLDisplay(QWidget* widget) {
    QMutexLocker lock(&s_registryMutex);
    s_displayRegistry.remove(widget);
}
