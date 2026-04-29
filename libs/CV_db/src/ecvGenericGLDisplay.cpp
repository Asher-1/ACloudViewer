// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericGLDisplay.h"

#include <ecvGuiParameters.h>

#include <QMap>
#include <QMutex>
#include <QMutexLocker>
#include <QWidget>

namespace {
QMutex s_registryMutex;
QMap<QWidget*, ecvGenericGLDisplay*> s_displayRegistry;
}  // namespace

// ================================================================
// Default implementations for the new per-view virtual methods.
// Concrete subclasses (ecvDisplayTools, ecvGLView) override these
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

int ecvGenericGLDisplay::glWidth() const {
    return getGLViewport().width();
}

int ecvGenericGLDisplay::glHeight() const {
    return getGLViewport().height();
}

int ecvGenericGLDisplay::getDevicePixelRatio() const {
    const QWidget* w = asWidget();
    return w ? static_cast<int>(w->devicePixelRatio()) : 1;
}

void ecvGenericGLDisplay::setInteractionMode(INTERACTION_FLAGS /*flags*/) {}

ecvGenericGLDisplay::INTERACTION_FLAGS
ecvGenericGLDisplay::getInteractionMode() const {
    return INTERACT_NONE;
}

void ecvGenericGLDisplay::setPickingMode(PICKING_MODE /*mode*/) {}

ecvGenericGLDisplay::PICKING_MODE
ecvGenericGLDisplay::getPickingMode() const {
    return NO_PICKING;
}

void ecvGenericGLDisplay::getContext(ccGLDrawContext& /*context*/) const {}

const ecvGui::ParamStruct& ecvGenericGLDisplay::getDisplayParameters() const {
    return ecvGui::Parameters();
}

void ecvGenericGLDisplay::setDisplayParameters(
        const ecvGui::ParamStruct& /*params*/, bool /*thisWindowOnly*/) {}

void ecvGenericGLDisplay::drawClickableItems(int /*xStart*/,
                                             int& /*yStart*/) {}

std::list<ccInteractor*>& ecvGenericGLDisplay::activeItemsRef() {
    static std::list<ccInteractor*> s_fallback;
    return s_fallback;
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
