// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAnimationParamDlg.h"

#include "ui_animationDlg.h"

// Local
#include "MainWindow.h"
#include "ecvFileUtils.h"
#include "ecvPickingHub.h"

// CV_CORE_LIB
#include <CVTools.h>
#include <ecvDisplayTools.h>

// Qt
#include <QApplication>
#include <QCheckBox>
#include <QDialogButtonBox>
#include <QMdiSubWindow>
#include <QProgressDialog>
#include <QPushButton>
#include <QtConcurrentRun>
#include <QtMath>

//=============================================================================
class AnimationDialogInternal : public Ui::AnimationParamDlg {
public:
    AnimationDialogInternal() {}

    ~AnimationDialogInternal() {}
};

ecvAnimationParamDlg::ecvAnimationParamDlg(QWidget* parent,
                                           MainWindow* app,
                                           ccPickingHub* pickingHub)
    : ccOverlayDialog(parent, Qt::Tool), m_app(app), m_pickingHub(pickingHub) {
    this->Internal = new AnimationDialogInternal();
    this->Internal->setupUi(this);

    this->Internal->EnablePickingAxis->setChecked(false);
    this->Internal->SavingViewports->setChecked(false);
    enablePickRotationAxis(false);

    connect(this->Internal->closeButtonBox, &QDialogButtonBox::clicked, this,
            &ecvAnimationParamDlg::onClose);
    QObject::connect(this->Internal->EnablePickingAxis, &QCheckBox::toggled,
                     this, &ecvAnimationParamDlg::enablePickRotationAxis);

    QObject::connect(this->Internal->angleButton, &QPushButton::clicked, this,
                     &ecvAnimationParamDlg::angleStep);

    QObject::connect(this->Internal->startAnimationButton,
                     &QPushButton::clicked, this,
                     &ecvAnimationParamDlg::startAnimation);
    QObject::connect(this->Internal->resetAnimationButton,
                     &QPushButton::clicked, this, &ecvAnimationParamDlg::reset);

    QObject::connect(this->Internal->pickingAxisStartToolButton,
                     &QToolButton::toggled, this,
                     &ecvAnimationParamDlg::updateAxisStartToolState);
    QObject::connect(this->Internal->pickingAxisEndToolButton,
                     &QToolButton::toggled, this,
                     &ecvAnimationParamDlg::updateAxisEndToolState);
}

ecvAnimationParamDlg::~ecvAnimationParamDlg() { delete this->Internal; }

double ecvAnimationParamDlg::getRotationAngle() const {
    return this->Internal->rotationAngle->value();
}

bool ecvAnimationParamDlg::isSavingViewport() const {
    return this->Internal->SavingViewports->isChecked();
}

void ecvAnimationParamDlg::startAnimation() {
    // we'll take the rendering time into account!
    QElapsedTimer timer;
    timer.start();

    // show progress dialog
    int viewport_num = 0;
    QProgressDialog progressDialog(
            QString("Saving Viewport number: %1").arg(viewport_num), "Cancel",
            0, 0, this);
    if (this->isSavingViewport()) {
        progressDialog.setWindowTitle("Rendering");
    } else {
        progressDialog.setWindowTitle("Preview");
    }

    progressDialog.show();
    progressDialog.setModal(false);
    progressDialog.setAutoClose(false);
    QApplication::processEvents();

    int fps = this->Internal->fpsSpinBox->value();
    // theoretical waiting time per frame
    qint64 delay_ms = static_cast<int>(1000 / fps);

    double angle_step = this->getRotationAngle();
    CCVector3d rotationAxis = this->getRotationAxis();
    CCVector2i pos(ecvDisplayTools::GlWidth() / 2,
                   ecvDisplayTools::GlHeight() / 2);

    while (true) {
        // next frame
        timer.restart();
        ecvDisplayTools::RotateWithAxis(pos, rotationAxis, angle_step, 0);
        if (this->isSavingViewport() && this->getMainWindow()) {
            this->getMainWindow()->doActionSaveViewportAsCamera();
            progressDialog.setLabelText(
                    QString("Saving Viewport number: %1").arg(++viewport_num));
        } else {
            progressDialog.setLabelText(QString("Render viewport to DB tree"));
        }

        progressDialog.setValue(viewport_num);
        progressDialog.update();
        QApplication::processEvents();
        if (progressDialog.wasCanceled()) {
            break;
        }

        qint64 dt_ms = timer.elapsed();

        // remaining time
        if (dt_ms < delay_ms) {
            int wait_ms = static_cast<int>(delay_ms - dt_ms);
            cloudViewer::utility::Sleep(wait_ms);
        }
    }
}

void ecvAnimationParamDlg::onItemPicked(const PickedItem& pi) {
    // with picking hub (CloudViewer)
    if (!m_associatedWin || !m_pickingHub) {
        assert(false);
        return;
    }

    AxisType axisType;
    if (this->Internal->pickingAxisStartToolButton->isChecked()) {
        axisType = AxisType::AXIS_START;
    } else if (this->Internal->pickingAxisEndToolButton->isChecked()) {
        axisType = AxisType::AXIS_END;
    }

    updateRotationAxisPoint(axisType, CCVector3d::fromArray(pi.P3D.u));

    if (axisType == AxisType::AXIS_START) {
        updateAxisStartToolState(false);
    } else if (axisType == AxisType::AXIS_END) {
        updateAxisEndToolState(false);
    }
}

void ecvAnimationParamDlg::processPickedItem(
        ccHObject* entity, unsigned, int, int, const CCVector3& P) {
    // without picking hub (ccViewer)
    if (!m_associatedWin) {
        assert(false);
        return;
    }

    if (!entity) {
        return;
    }

    AxisType axisType;
    if (this->Internal->pickingAxisStartToolButton->isChecked()) {
        axisType = AxisType::AXIS_START;
    } else if (this->Internal->pickingAxisEndToolButton->isChecked()) {
        axisType = AxisType::AXIS_END;
    }

    updateRotationAxisPoint(axisType, CCVector3d::fromArray(P.u));

    if (axisType == AxisType::AXIS_START) {
        updateAxisStartToolState(false);
    } else if (axisType == AxisType::AXIS_END) {
        updateAxisEndToolState(false);
    }
}

bool ecvAnimationParamDlg::start() {
    ccOverlayDialog::start();

    // no such concept for this dialog!
    // (+ we want to allow dynamic change of associated window)
    m_processing = false;

    // cache history viewport params
    viewportParamsHistory = ecvDisplayTools::GetViewportParameters();

    return true;
}

void ecvAnimationParamDlg::linkWith(QMdiSubWindow* qWin) {
    // corresponding ccGLWindow
    QWidget* associatedWin =
            (qWin ? static_cast<QWidget*>(qWin->widget()) : nullptr);

    linkWith(associatedWin);
}

bool ecvAnimationParamDlg::linkWith(QWidget* win) {
    QWidget* oldWin = m_associatedWin;

    if (!ccOverlayDialog::linkWith(win)) {
        return false;
    }

    if (oldWin != m_associatedWin) {
        // automatically disable picking mode when changing th
        if (this->Internal->pickingAxisStartToolButton->isChecked()) {
            updateAxisStartToolState(false);
        }

        if (this->Internal->pickingAxisEndToolButton->isChecked()) {
            updateAxisEndToolState(false);
        }
    }

    if (oldWin) {
        oldWin->disconnect(this);
    }

    if (m_associatedWin) {
        initWith(m_associatedWin);
        connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::destroyed,
                this, &QWidget::hide);
    }

    return true;
}

void ecvAnimationParamDlg::initWith(QWidget* win) {
    setEnabled(win != nullptr);
    if (!win) return;
}

CCVector3d ecvAnimationParamDlg::getRotationAxis() const {
    if (this->Internal->EnablePickingAxis->isChecked()) {
        CCVector3d axisStart(this->Internal->axisStartXDoubleSpinBox->value(),
                             this->Internal->axisStartYDoubleSpinBox->value(),
                             this->Internal->axisStartZDoubleSpinBox->value());
        CCVector3d axisEnd(this->Internal->axisEndXDoubleSpinBox->value(),
                           this->Internal->axisEndYDoubleSpinBox->value(),
                           this->Internal->axisEndZDoubleSpinBox->value());
        return axisEnd - axisStart;
    }

    return CCVector3d(this->Internal->axisXDoubleSpinBox->value(),
                      this->Internal->axisYDoubleSpinBox->value(),
                      this->Internal->axisZDoubleSpinBox->value());
}

void ecvAnimationParamDlg::updateRotationAxisPoint(AxisType axisType,
                                                   const CCVector3d& P) {
    if (AxisType::AXIS_START == axisType) {
        this->Internal->axisStartXDoubleSpinBox->blockSignals(true);
        this->Internal->axisStartYDoubleSpinBox->blockSignals(true);
        this->Internal->axisStartZDoubleSpinBox->blockSignals(true);
        this->Internal->axisStartXDoubleSpinBox->setValue(P.x);
        this->Internal->axisStartYDoubleSpinBox->setValue(P.y);
        this->Internal->axisStartZDoubleSpinBox->setValue(P.z);
        this->Internal->axisStartXDoubleSpinBox->blockSignals(false);
        this->Internal->axisStartYDoubleSpinBox->blockSignals(false);
        this->Internal->axisStartZDoubleSpinBox->blockSignals(false);
    } else if (AxisType::AXIS_END == axisType) {
        this->Internal->axisEndXDoubleSpinBox->blockSignals(true);
        this->Internal->axisEndYDoubleSpinBox->blockSignals(true);
        this->Internal->axisEndZDoubleSpinBox->blockSignals(true);
        this->Internal->axisEndXDoubleSpinBox->setValue(P.x);
        this->Internal->axisEndYDoubleSpinBox->setValue(P.y);
        this->Internal->axisEndZDoubleSpinBox->setValue(P.z);
        this->Internal->axisEndXDoubleSpinBox->blockSignals(false);
        this->Internal->axisEndYDoubleSpinBox->blockSignals(false);
        this->Internal->axisEndZDoubleSpinBox->blockSignals(false);
    }
}

void ecvAnimationParamDlg::reset() {
    ecvDisplayTools::SetViewportParameters(viewportParamsHistory);
    ecvDisplayTools::UpdateScreen();
}

//-----------------------------------------------------------------------------
void ecvAnimationParamDlg::enablePickRotationAxis(bool state) {
    auto& internal = (*this->Internal);
    internal.pickingAxisStartToolButton->setEnabled(state);
    internal.axisStartXDoubleSpinBox->setEnabled(state);
    internal.axisStartYDoubleSpinBox->setEnabled(state);
    internal.axisStartZDoubleSpinBox->setEnabled(state);
    internal.pickingAxisEndToolButton->setEnabled(state);
    internal.axisEndXDoubleSpinBox->setEnabled(state);
    internal.axisEndYDoubleSpinBox->setEnabled(state);
    internal.axisEndZDoubleSpinBox->setEnabled(state);
    internal.axisXDoubleSpinBox->setEnabled(!state);
    internal.axisYDoubleSpinBox->setEnabled(!state);
    internal.axisZDoubleSpinBox->setEnabled(!state);
}

//-----------------------------------------------------------------------------
void ecvAnimationParamDlg::angleStep() {
    double angle_step = this->getRotationAngle();
    CCVector2i pos(ecvDisplayTools::GlWidth() / 2,
                   ecvDisplayTools::GlHeight() / 2);
    CCVector3d rotationAxis = getRotationAxis();
    ecvDisplayTools::RotateWithAxis(pos, rotationAxis, angle_step, 0);
    if (this->isSavingViewport() && this->getMainWindow()) {
        this->getMainWindow()->doActionSaveViewportAsCamera();
    }
}

void ecvAnimationParamDlg::enableListener(bool state) {
    if (m_pickingHub) {
        if (state) {
            if (!m_pickingHub->addListener(this, true)) {
                CVLog::Error(
                        "Can't start the picking process (another tool is "
                        "using it)");
                state = false;
            }
        } else {
            m_pickingHub->removeListener(this);
        }
    } else if (m_associatedWin) {
        if (state) {
            ecvDisplayTools::SetPickingMode(
                    ecvDisplayTools::POINT_OR_TRIANGLE_PICKING);
            connect(ecvDisplayTools::TheInstance(),
                    &ecvDisplayTools::itemPicked, this,
                    &ecvAnimationParamDlg::processPickedItem);
        } else {
            ecvDisplayTools::SetPickingMode(ecvDisplayTools::DEFAULT_PICKING);
            disconnect(ecvDisplayTools::TheInstance(),
                       &ecvDisplayTools::itemPicked, this,
                       &ecvAnimationParamDlg::processPickedItem);
        }
    }
}

void ecvAnimationParamDlg::updateAxisStartToolState(bool state) {
    enableListener(state);
    this->Internal->pickingAxisStartToolButton->blockSignals(true);
    this->Internal->pickingAxisStartToolButton->setChecked(state);
    this->Internal->pickingAxisStartToolButton->blockSignals(false);
}

void ecvAnimationParamDlg::updateAxisEndToolState(bool state) {
    enableListener(state);
    this->Internal->pickingAxisEndToolButton->blockSignals(true);
    this->Internal->pickingAxisEndToolButton->setChecked(state);
    this->Internal->pickingAxisEndToolButton->blockSignals(false);
}
