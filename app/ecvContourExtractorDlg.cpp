// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvContourExtractorDlg.h"

// CV_CORE_LIB
#include <CVPlatform.h>

// CV_DB_LIB
#include <ecvGenericGLDisplay.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>
#include <ecvViewportParameters.h>

// Qt
#include <QCoreApplication>
#include <QRect>

// system
#include <assert.h>
#if defined(CV_WINDOWS)
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>

#include <algorithm>
#endif

ccContourExtractorDlg::ccContourExtractorDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::WindowMaximizeButtonHint | Qt::WindowCloseButtonHint),
      Ui::ContourExtractorDlg(),
      m_skipped(false) {
    setupUi(this);
}

void ccContourExtractorDlg::init() {
    connect(nextPushButton, &QAbstractButton::clicked, &m_loop,
            &QEventLoop::quit);
    // connect(nextPushButton, SIGNAL(clicked()), this, SLOT(accept()));
    connect(skipPushButton, &QAbstractButton::clicked, this,
            &ccContourExtractorDlg::onSkipButtonClicked);
    nextPushButton->setFocus();

    // create 3D window
    {
        // QWidget* glWidget = 0;
        // CreateGLWindow(m_glWindow, glWidget, false, true);
        // assert(m_glWindow && glWidget);

        if (auto* view = ecvViewManager::instance().getEffectiveView()) {
            ecvGui::ParamStruct params = view->getDisplayParameters();
            // black (text) & white (background) display by default
            params.backgroundCol = ecvColor::white;
            params.textDefaultCol = ecvColor::black;
            params.pointsDefaultCol = ecvColor::black;
            params.drawBackgroundGradient = false;
            params.decimateMeshOnMove = false;
            params.displayCross = false;
            params.colorScaleUseShader = false;
            view->setDisplayParameters(params);
            view->setPerspectiveState(false, true);
            view->setInteractionMode(
                    ecvGenericGLDisplay::INTERACT_PAN |
                    ecvGenericGLDisplay::INTERACT_ZOOM_CAMERA |
                    ecvGenericGLDisplay::INTERACT_CLICKABLE_ITEMS);
            view->setPickingMode(ecvGenericGLDisplay::NO_PICKING);
            if (auto* ctx = view->viewContext())
                ctx->displayOverlayEntities = true;
            viewFrame->setLayout(new QHBoxLayout);
            viewFrame->layout()->addWidget(MainWindow::TheInstance());
        }
    }
}

void ccContourExtractorDlg::zoomOn(const ccBBox& box) {
    QRect screenRect;
    if (auto* w = ecvViewManager::instance().activeWidget()) {
        screenRect = w->geometry();
        const QPoint gp = w->mapToGlobal(screenRect.topLeft());
        screenRect.setTopLeft(gp);
    }
    const float pixSize = std::max(
            box.getDiagVec().x / std::max(20, screenRect.width() - 20),
            box.getDiagVec().y / std::max(20, screenRect.height() - 20));

    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        ecvViewportParameters params = view->getViewportParameters();
        params.pixelSize = pixSize;
        params.setCameraCenter(CCVector3d::fromArray(box.getCenter().u), true);
        view->setViewportParameters(params);
        view->invalidateViewport();
        view->deprecate3DLayer();
    }
}

bool ccContourExtractorDlg::isSkipped() const {
    return skipPushButton->isChecked();
}

void ccContourExtractorDlg::addToDisplay(ccHObject* obj,
                                         bool noDependency /*=true*/) {
    if (obj) {
        if (auto* view = ecvViewManager::instance().getEffectiveView())
            view->addToOwnDB(obj, noDependency);
    }
}

void ccContourExtractorDlg::removFromDisplay(ccHObject* obj) {
    if (obj) {
        if (auto* view = ecvViewManager::instance().getEffectiveView())
            view->removeFromOwnDB(obj);
    }
}

void ccContourExtractorDlg::refresh() {
    if (m_skipped) return;
    { ecvRedrawScope scope; }
    QCoreApplication::processEvents();
}

void ccContourExtractorDlg::displayMessage(
        QString message, bool waitForUserConfirmation /*=false*/) {
    if (m_skipped) return;
    messageLabel->setText(message);
    if (waitForUserConfirmation) waitForUser(20);
}

void ccContourExtractorDlg::onSkipButtonClicked() {
    m_skipped = true;
    hide();
    QCoreApplication::processEvents();
}

void ccContourExtractorDlg::waitForUser(unsigned defaultDelay_ms /*=100*/) {
    if (m_skipped) return;

    if (autoCheckBox->isChecked()) {
        // simply wait a pre-determined time
#if defined(CV_WINDOWS)
        ::Sleep(defaultDelay_ms);
#else
        usleep(defaultDelay_ms * 1000);
#endif
    } else {
        setModal(true);
        // wait for the user to click on the 'Next' button
        m_loop.exec();
        setModal(false);
        // exec();
    }
}
