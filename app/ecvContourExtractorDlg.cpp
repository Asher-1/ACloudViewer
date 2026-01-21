// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvContourExtractorDlg.h"

// CV_CORE_LIB
#include <CVPlatform.h>

// Qt
#include <QCoreApplication>

// system
#include <assert.h>
#if defined(CV_WINDOWS)
#include <windows.h>
#else
#include <time.h>
#include <unistd.h>
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

        ecvGui::ParamStruct params = ecvDisplayTools::GetDisplayParameters();
        // black (text) & white (background) display by default
        params.backgroundCol = ecvColor::white;
        params.textDefaultCol = ecvColor::black;
        params.pointsDefaultCol = ecvColor::black;
        params.drawBackgroundGradient = false;
        params.decimateMeshOnMove = false;
        params.displayCross = false;
        params.colorScaleUseShader = false;
        ecvDisplayTools::SetDisplayParameters(params);
        ecvDisplayTools::SetPerspectiveState(false, true);
        ecvDisplayTools::SetInteractionMode(
                ecvDisplayTools::INTERACT_PAN |
                ecvDisplayTools::INTERACT_ZOOM_CAMERA |
                ecvDisplayTools::INTERACT_CLICKABLE_ITEMS);
        ecvDisplayTools::SetPickingMode(ecvDisplayTools::NO_PICKING);
        ecvDisplayTools::DisplayOverlayEntities(true);
        viewFrame->setLayout(new QHBoxLayout);
        viewFrame->layout()->addWidget(ecvDisplayTools::GetMainWindow());
    }
}

void ccContourExtractorDlg::zoomOn(const ccBBox& box) {
    float pixSize = std::max(
            box.getDiagVec().x /
                    std::max(20, ecvDisplayTools::GetScreenRect().width() - 20),
            box.getDiagVec().y /
                    std::max(20,
                             ecvDisplayTools::GetScreenRect().height() - 20));
    ecvDisplayTools::SetPixelSize(pixSize);
    ecvDisplayTools::SetCameraPos(CCVector3d::fromArray(box.getCenter().u));
}

bool ccContourExtractorDlg::isSkipped() const {
    return skipPushButton->isChecked();
}

void ccContourExtractorDlg::addToDisplay(ccHObject* obj,
                                         bool noDependency /*=true*/) {
    if (obj) {
        ecvDisplayTools::AddToOwnDB(obj, noDependency);
    }
}

void ccContourExtractorDlg::removFromDisplay(ccHObject* obj) {
    if (obj) {
        ecvDisplayTools::RemoveFromOwnDB(obj);
    }
}

void ccContourExtractorDlg::refresh() {
    if (m_skipped) return;
    ecvDisplayTools::RedrawDisplay();
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
