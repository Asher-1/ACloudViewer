// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvCameraParamEditDlg.h"

#include "ui_cameraParamDlg.h"

// Local
#include "ecvCustomViewpointButtonDlg.h"
#include "ecvFileUtils.h"
#include "ecvPersistentSettings.h"
#include "ecvPickingHub.h"
#include "ecvSettingManager.h"

// CV_DB_LIB
#include <ecvGenericCameraTool.h>

// CV_CORE_LIB
#include <CVConst.h>
#include <CVTools.h>
#include <GenericTriangle.h>

// Qt
#include <QFileDialog>
#include <QMdiSubWindow>
#include <QtMath>

namespace {
QStringList getListOfStrings(ecvSettingManager* settings,
                             const QString& defaultTxt,
                             int min,
                             int max) {
    QStringList val;
    for (int cc = 0; cc < max; ++cc) {
        const QString key = QString::number(cc);
        if (cc < min || settings->contains(key)) {
            val << settings->value(key, defaultTxt).toString();
        } else {
            break;
        }
    }
    return val;
}
}  // namespace

//=============================================================================
class CameraDialogInternal : public Ui::CameraParamDlg {
    QVector<QPointer<QToolButton>> CustomViewpointButtons;
    QPointer<QToolButton> PlusButton;

public:
    CameraDialogInternal() {
        // Add + bouton
        this->PlusButton = new QToolButton();
        this->PlusButton->setObjectName("AddButton");
        this->PlusButton->setToolTip(QToolButton::tr("Add Current Viewpoint"));
        this->PlusButton->setIcon(QIcon(":/Resources/images/svg/pqPlus.png"));
        this->PlusButton->setMinimumSize(QSize(34, 34));
    }

    ~CameraDialogInternal() { delete this->PlusButton; }

    void updateCustomViewpointButtons(::ecvCameraParamEditDlg* self) {
        QStringList toolTips = self->CustomViewpointToolTips();

        // Remove supplemental buttons
        this->PlusButton->disconnect();
        this->customViewpointGridLayout->removeWidget(this->PlusButton);

        for (int cc = this->CustomViewpointButtons.size(); cc > toolTips.size();
             cc--) {
            this->customViewpointGridLayout->removeWidget(
                    this->CustomViewpointButtons[cc - 1]);
            delete this->CustomViewpointButtons[cc - 1];
        }
        if (this->CustomViewpointButtons.size() > toolTips.size()) {
            this->CustomViewpointButtons.resize(toolTips.size());
        }

        // add / change remaining buttons.
        for (int cc = 0; cc < toolTips.size(); ++cc) {
            if (this->CustomViewpointButtons.size() > cc) {
                this->CustomViewpointButtons[cc]->setToolTip(toolTips[cc]);
            } else {
                QToolButton* tb = new QToolButton(self);
                tb->setObjectName(QString("customViewpoint%1").arg(cc));
                tb->setText(QString::number(cc + 1));
                tb->setToolTip(toolTips[cc]);
                tb->setProperty("pqCameraDialog_INDEX", cc);
                tb->setMinimumSize(QSize(34, 34));
                self->connect(tb, SIGNAL(clicked()),
                              SLOT(ApplyCustomViewpoint()));
                this->CustomViewpointButtons.push_back(tb);
                this->customViewpointGridLayout->addWidget(tb, cc / 6, cc % 6);
            }
        }

        // Add Plus Button if needed
        if (toolTips.size() <
            ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS) {
            self->connect(this->PlusButton, SIGNAL(clicked()),
                          SLOT(addCurrentViewpointToCustomViewpoints()));
            this->customViewpointGridLayout->addWidget(
                    this->PlusButton, toolTips.size() / 6, toolTips.size() % 6);
        }
    }
};

ecvCameraParamEditDlg::ecvCameraParamEditDlg(QWidget* parent,
                                             ccPickingHub* pickingHub)
    //: ccOverlayDialog(parent, pickingHub ? Qt::FramelessWindowHint | Qt::Tool
    //: : Qt::Tool) //pickingHub = ACloudViewer / otherwise = ccViewer
    : ccOverlayDialog(parent, Qt::Tool), m_pickingHub(pickingHub) {
    this->Internal = new CameraDialogInternal;
    this->Internal->setupUi(this);
    QObject::connect(this->Internal->viewXPlus, SIGNAL(clicked()), this,
                     SLOT(resetViewDirectionPosX()));
    QObject::connect(this->Internal->viewXMinus, SIGNAL(clicked()), this,
                     SLOT(resetViewDirectionNegX()));
    QObject::connect(this->Internal->viewYPlus, SIGNAL(clicked()), this,
                     SLOT(resetViewDirectionPosY()));
    QObject::connect(this->Internal->viewYMinus, SIGNAL(clicked()), this,
                     SLOT(resetViewDirectionNegY()));
    QObject::connect(this->Internal->viewZPlus, SIGNAL(clicked()), this,
                     SLOT(resetViewDirectionPosZ()));
    QObject::connect(this->Internal->viewZMinus, SIGNAL(clicked()), this,
                     SLOT(resetViewDirectionNegZ()));

    QObject::connect(this->Internal->AutoPickCenterOfRotation,
                     SIGNAL(toggled(bool)), this,
                     SLOT(autoPickRotationCenterWithCamera()));

    QObject::connect(this->Internal->rollButton, SIGNAL(clicked()), this,
                     SLOT(applyCameraRoll()));
    QObject::connect(this->Internal->elevationButton, SIGNAL(clicked()), this,
                     SLOT(applyCameraElevation()));
    QObject::connect(this->Internal->azimuthButton, SIGNAL(clicked()), this,
                     SLOT(applyCameraAzimuth()));
    QObject::connect(this->Internal->zoomInButton, SIGNAL(clicked()), this,
                     SLOT(applyCameraZoomIn()));
    QObject::connect(this->Internal->zoomOutButton, SIGNAL(clicked()), this,
                     SLOT(applyCameraZoomOut()));

    QObject::connect(this->Internal->saveCameraConfiguration, SIGNAL(clicked()),
                     this, SLOT(saveCameraConfiguration()));

    QObject::connect(this->Internal->loadCameraConfiguration, SIGNAL(clicked()),
                     this, SLOT(loadCameraConfiguration()));

    QObject::connect(this->Internal->rcxDoubleSpinBox,
                     static_cast<void (QDoubleSpinBox::*)(double)>(
                             &QDoubleSpinBox::valueChanged),
                     this, &ecvCameraParamEditDlg::pivotChanged);
    QObject::connect(this->Internal->rcyDoubleSpinBox,
                     static_cast<void (QDoubleSpinBox::*)(double)>(
                             &QDoubleSpinBox::valueChanged),
                     this, &ecvCameraParamEditDlg::pivotChanged);
    QObject::connect(this->Internal->rczDoubleSpinBox,
                     static_cast<void (QDoubleSpinBox::*)(double)>(
                             &QDoubleSpinBox::valueChanged),
                     this, &ecvCameraParamEditDlg::pivotChanged);

    QObject::connect(this->Internal->rotationFactor,
                     static_cast<void (QDoubleSpinBox::*)(double)>(
                             &QDoubleSpinBox::valueChanged),
                     this, &ecvCameraParamEditDlg::rotationFactorChanged);
    QObject::connect(this->Internal->factorHorizontalSlider,
                     &QAbstractSlider::valueChanged, this,
                     &ecvCameraParamEditDlg::zfactorSliderMoved);

    QObject::connect(this->Internal->configureCustomViewpoints,
                     SIGNAL(clicked()), this,
                     SLOT(ConfigureCustomViewpoints()));

    this->Internal->AutoPickCenterOfRotation->setChecked(
            ecvDisplayTools::AutoPickPivotAtCenter());

    QObject::connect(this->Internal->updatePushButton, SIGNAL(clicked()), this,
                     SLOT(updateCamera()));
    QObject::connect(this->Internal->pivotPickingToolButton,
                     &QAbstractButton::toggled, this,
                     &ecvCameraParamEditDlg::pickPointAsPivot);
    QObject::connect(ecvSettingManager::TheInstance(), SIGNAL(modified()),
                     SLOT(updateCustomViewpointButtons()));
    QObject::connect(ecvDisplayTools::TheInstance(),
                     &ecvDisplayTools::cameraParamChanged, this,
                     &ecvCameraParamEditDlg::cameraChanged);

    // load custom view buttons with any tool tips set by the user in a previous
    // session.
    this->updateCustomViewpointButtons();
}

ecvCameraParamEditDlg::~ecvCameraParamEditDlg() {
    delete this->Internal;
    if (m_tool) {
        delete m_tool;
        m_tool = nullptr;
    }
}

bool ecvCameraParamEditDlg::setCameraTool(ecvGenericCameraTool* tool) {
    if (!tool) return false;

    m_tool = tool;
    return true;
}

void ecvCameraParamEditDlg::onItemPicked(const PickedItem& pi) {
    // with picking hub (CloudViewer)
    if (!m_associatedWin || !m_pickingHub) {
        assert(false);
        return;
    }

    ecvDisplayTools::SetPivotPoint(CCVector3d::fromArray(pi.P3D.u));
    ecvDisplayTools::UpdateScreen();

    pickPointAsPivot(false);
}

void ecvCameraParamEditDlg::processPickedItem(
        ccHObject* entity, unsigned, int, int, const CCVector3& P) {
    // without picking hub (ccViewer)
    if (!m_associatedWin) {
        assert(false);
        return;
    }

    if (!entity) {
        return;
    }

    ecvDisplayTools::SetPivotPoint(CCVector3d::fromArray(P.u));
    ecvDisplayTools::UpdateScreen();

    pickPointAsPivot(false);
}

bool ecvCameraParamEditDlg::start() {
    ccOverlayDialog::start();

    // no such concept for this dialog!
    // (+ we want to allow dynamic change of associated window)
    m_processing = false;

    return true;
}

void ecvCameraParamEditDlg::linkWith(QMdiSubWindow* qWin) {
    // corresponding MainWindow
    QWidget* associatedWin =
            (qWin ? static_cast<QWidget*>(qWin->widget()) : nullptr);

    linkWith(associatedWin);
}

bool ecvCameraParamEditDlg::linkWith(QWidget* win) {
    QWidget* oldWin = m_associatedWin;

    if (!ccOverlayDialog::linkWith(win)) {
        return false;
    }

    if (oldWin != m_associatedWin &&
        this->Internal->pivotPickingToolButton->isChecked()) {
        // automatically disable picking mode when changing th
        pickPointAsPivot(false);
    }

    if (oldWin) {
        oldWin->disconnect(this);
    }

    if (m_associatedWin) {
        initWith(m_associatedWin);
        connect(ecvDisplayTools::TheInstance(),
                &ecvDisplayTools::pivotPointChanged, this,
                &ecvCameraParamEditDlg::updatePivotPoint);
        connect(ecvDisplayTools::TheInstance(),
                &ecvDisplayTools::perspectiveStateChanged, this,
                &ecvCameraParamEditDlg::updateViewMode);
        connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::destroyed,
                this, &QWidget::hide);
    }

    return true;
}

void ecvCameraParamEditDlg::initWith(QWidget* win) {
    setEnabled(win != nullptr);
    if (!win) return;

    // update view mode
    updateViewMode();

    if (m_tool) {
        m_tool->updateCameraParameters();
        this->updateUi();
    }
}

void ecvCameraParamEditDlg::reflectParamChange() {
    if (!m_tool) return;

    m_tool->updateCamera();
}

void ecvCameraParamEditDlg::updatePivotPoint(const CCVector3d& P) {
    this->Internal->rcxDoubleSpinBox->blockSignals(true);
    this->Internal->rcyDoubleSpinBox->blockSignals(true);
    this->Internal->rczDoubleSpinBox->blockSignals(true);
    this->Internal->rcxDoubleSpinBox->setValue(P.x);
    this->Internal->rcyDoubleSpinBox->setValue(P.y);
    this->Internal->rczDoubleSpinBox->setValue(P.z);
    this->Internal->rcxDoubleSpinBox->blockSignals(false);
    this->Internal->rcyDoubleSpinBox->blockSignals(false);
    this->Internal->rczDoubleSpinBox->blockSignals(false);
}

void ecvCameraParamEditDlg::updateViewMode() {
    if (m_associatedWin) {
        bool objectBased = true;
        bool perspective = ecvDisplayTools::GetPerspectiveState();

        if (!perspective) {
            this->Internal->currentModeLabel->setText("parallel projection");
        } else {
            this->Internal->currentModeLabel->setText("perspective projection");
        }

        this->Internal->pivotPickingToolButton->setEnabled(objectBased);
        this->Internal->eyeAngle->setEnabled(perspective);
    }
}

void ecvCameraParamEditDlg::updateCamera() {
    if (!m_tool) return;

    m_tool->CurrentCameraParam.position.x = this->Internal->xPosition->value();
    m_tool->CurrentCameraParam.position.y = this->Internal->zPosition->value();
    m_tool->CurrentCameraParam.position.z = this->Internal->yPosition->value();

    m_tool->CurrentCameraParam.focal.x = this->Internal->xFocal->value();
    m_tool->CurrentCameraParam.focal.y = this->Internal->yFocal->value();
    m_tool->CurrentCameraParam.focal.z = this->Internal->zFocal->value();

    m_tool->CurrentCameraParam.viewUp.x = this->Internal->xViewup->value();
    m_tool->CurrentCameraParam.viewUp.y = this->Internal->yViewup->value();
    m_tool->CurrentCameraParam.viewUp.z = this->Internal->zViewup->value();

    m_tool->CurrentCameraParam.rotationFactor =
            this->Internal->rotationFactor->value();
    m_tool->CurrentCameraParam.viewAngle = this->Internal->viewAngle->value();
    m_tool->CurrentCameraParam.eyeAngle = this->Internal->eyeAngle->value();
    m_tool->CurrentCameraParam.clippRange.x =
            this->Internal->nearClipping->value();
    m_tool->CurrentCameraParam.clippRange.y =
            this->Internal->farClipping->value();

    reflectParamChange();
}

void ecvCameraParamEditDlg::cameraChanged() {
    if (!this->m_tool) return;

    this->m_tool->updateCameraParameters();
    this->updateUi();
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::autoPickRotationCenterWithCamera() {
    ecvDisplayTools::SendAutoPickPivotAtCenter(
            this->Internal->AutoPickCenterOfRotation->isChecked());
    // m_app->setAutoPickPivot(this->Internal->AutoPickCenterOfRotation->isChecked());
}

void ecvCameraParamEditDlg::updateUi() {
    if (!m_tool) return;

    this->Internal->xPosition->blockSignals(true);
    this->Internal->zPosition->blockSignals(true);
    this->Internal->yPosition->blockSignals(true);
    this->Internal->xPosition->setValue(m_tool->CurrentCameraParam.position.x);
    this->Internal->zPosition->setValue(m_tool->CurrentCameraParam.position.y);
    this->Internal->yPosition->setValue(m_tool->CurrentCameraParam.position.z);
    this->Internal->xPosition->blockSignals(false);
    this->Internal->zPosition->blockSignals(false);
    this->Internal->yPosition->blockSignals(false);

    this->Internal->xFocal->blockSignals(true);
    this->Internal->yFocal->blockSignals(true);
    this->Internal->zFocal->blockSignals(true);
    this->Internal->xFocal->setValue(m_tool->CurrentCameraParam.focal.x);
    this->Internal->yFocal->setValue(m_tool->CurrentCameraParam.focal.y);
    this->Internal->zFocal->setValue(m_tool->CurrentCameraParam.focal.z);
    this->Internal->xFocal->blockSignals(false);
    this->Internal->yFocal->blockSignals(false);
    this->Internal->zFocal->blockSignals(false);

    this->Internal->xViewup->blockSignals(true);
    this->Internal->yViewup->blockSignals(true);
    this->Internal->zViewup->blockSignals(true);
    this->Internal->xViewup->setValue(m_tool->CurrentCameraParam.viewUp.x);
    this->Internal->yViewup->setValue(m_tool->CurrentCameraParam.viewUp.y);
    this->Internal->zViewup->setValue(m_tool->CurrentCameraParam.viewUp.z);
    this->Internal->xViewup->blockSignals(false);
    this->Internal->yViewup->blockSignals(false);
    this->Internal->zViewup->blockSignals(false);

    this->Internal->nearClipping->blockSignals(true);
    this->Internal->farClipping->blockSignals(true);
    this->Internal->nearClipping->setValue(
            m_tool->CurrentCameraParam.clippRange.x);
    this->Internal->farClipping->setValue(
            m_tool->CurrentCameraParam.clippRange.y);
    this->Internal->nearClipping->blockSignals(false);
    this->Internal->farClipping->blockSignals(false);

    this->Internal->viewAngle->blockSignals(true);
    this->Internal->viewAngle->setValue(m_tool->CurrentCameraParam.viewAngle);
    this->Internal->viewAngle->blockSignals(false);

    this->Internal->eyeAngle->blockSignals(true);
    this->Internal->eyeAngle->setValue(m_tool->CurrentCameraParam.eyeAngle);
    this->Internal->eyeAngle->blockSignals(false);

    this->Internal->rcxDoubleSpinBox->blockSignals(true);
    this->Internal->rcyDoubleSpinBox->blockSignals(true);
    this->Internal->rczDoubleSpinBox->blockSignals(true);
    this->Internal->rcxDoubleSpinBox->setValue(
            m_tool->CurrentCameraParam.pivot.x);
    this->Internal->rcyDoubleSpinBox->setValue(
            m_tool->CurrentCameraParam.pivot.y);
    this->Internal->rczDoubleSpinBox->setValue(
            m_tool->CurrentCameraParam.pivot.z);
    this->Internal->rcxDoubleSpinBox->blockSignals(false);
    this->Internal->rcyDoubleSpinBox->blockSignals(false);
    this->Internal->rczDoubleSpinBox->blockSignals(false);

    this->Internal->factorHorizontalSlider->blockSignals(true);
    this->Internal->factorHorizontalSlider->setValue(
            qFloor(m_tool->CurrentCameraParam.rotationFactor * 10.0));
    this->Internal->factorHorizontalSlider->blockSignals(false);

    this->Internal->rotationFactor->blockSignals(true);
    this->Internal->rotationFactor->setValue(
            m_tool->CurrentCameraParam.rotationFactor);
    this->Internal->rotationFactor->blockSignals(false);
}

void ecvCameraParamEditDlg::pivotChanged() {
    if (!m_associatedWin) return;

    m_associatedWin->blockSignals(true);
    m_tool->CurrentCameraParam.pivot =
            CCVector3d(this->Internal->rcxDoubleSpinBox->value(),
                       this->Internal->rcyDoubleSpinBox->value(),
                       this->Internal->rczDoubleSpinBox->value());
    m_associatedWin->blockSignals(false);

    reflectParamChange();
}

void ecvCameraParamEditDlg::rotationFactorChanged(double val) {
    this->Internal->factorHorizontalSlider->blockSignals(true);
    this->Internal->factorHorizontalSlider->setValue(qFloor(val * 10.0));
    this->Internal->factorHorizontalSlider->blockSignals(false);

    m_tool->CurrentCameraParam.rotationFactor =
            this->Internal->rotationFactor->value();
    reflectParamChange();
}

void ecvCameraParamEditDlg::zfactorSliderMoved(int val) {
    this->Internal->rotationFactor->setValue(val / 10.0);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::SetCameraGroupsEnabled(bool enabled) {
    auto& internal = (*this->Internal);
    internal.viewXMinus->setEnabled(enabled);
    internal.viewXPlus->setEnabled(enabled);
    internal.viewYMinus->setEnabled(enabled);
    internal.viewYPlus->setEnabled(enabled);
    internal.viewZMinus->setEnabled(enabled);
    internal.viewZPlus->setEnabled(enabled);

    internal.customViewpointGridLayout->setEnabled(enabled);
    internal.configureCustomViewpoints->setEnabled(enabled);

    internal.rcxDoubleSpinBox->setEnabled(enabled);
    internal.rcyDoubleSpinBox->setEnabled(enabled);
    internal.rczDoubleSpinBox->setEnabled(enabled);
    internal.AutoPickCenterOfRotation->setEnabled(enabled);

    internal.rotationFactor->setEnabled(enabled);

    internal.xPosition->setEnabled(enabled);
    internal.yPosition->setEnabled(enabled);
    internal.zPosition->setEnabled(enabled);
    internal.xFocal->setEnabled(enabled);
    internal.yFocal->setEnabled(enabled);
    internal.zFocal->setEnabled(enabled);
    internal.xViewup->setEnabled(enabled);
    internal.yViewup->setEnabled(enabled);
    internal.zViewup->setEnabled(enabled);
    internal.viewAngle->setEnabled(enabled);
    internal.nearClipping->setEnabled(enabled);
    internal.farClipping->setEnabled(enabled);
    internal.loadCameraConfiguration->setEnabled(enabled);
    internal.saveCameraConfiguration->setEnabled(enabled);
    internal.updatePushButton->setEnabled(enabled);

    internal.rollButton->setEnabled(enabled);
    internal.rollAngle->setEnabled(enabled);
    internal.elevationButton->setEnabled(enabled);
    internal.elevationAngle->setEnabled(enabled);
    internal.azimuthButton->setEnabled(enabled);
    internal.azimuthAngle->setEnabled(enabled);
    internal.zoomInButton->setEnabled(enabled);
    internal.zoomFactor->setEnabled(enabled);
    internal.zoomOutButton->setEnabled(enabled);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirection(double look_x,
                                               double look_y,
                                               double look_z,
                                               double up_x,
                                               double up_y,
                                               double up_z) {
    if (this->m_tool) {
        this->m_tool->resetViewDirection(look_x, look_y, look_z, up_x, up_y,
                                         up_z);
        this->cameraChanged();
    }
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirectionPosX() {
    this->resetViewDirection(1, 0, 0, 0, 0, 1);
}
//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirectionNegX() {
    this->resetViewDirection(-1, 0, 0, 0, 0, 1);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirectionPosY() {
    this->resetViewDirection(0, 1, 0, 0, 0, 1);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirectionNegY() {
    this->resetViewDirection(0, -1, 0, 0, 0, 1);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirectionPosZ() {
    this->resetViewDirection(0, 0, 1, 0, 1, 0);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::resetViewDirectionNegZ() {
    this->resetViewDirection(0, 0, -1, 0, 1, 0);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::adjustCamera(CameraAdjustmentType enType,
                                         double value) {
    if (this->m_tool) {
        this->m_tool->adjustCamera(
                ecvGenericCameraTool::CameraAdjustmentType(enType), value);
        this->cameraChanged();
    }
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::applyCameraRoll() {
    this->adjustCamera(CameraAdjustmentType::Roll,
                       this->Internal->rollAngle->value());
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::applyCameraElevation() {
    this->adjustCamera(CameraAdjustmentType::Elevation,
                       this->Internal->elevationAngle->value());
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::applyCameraAzimuth() {
    this->adjustCamera(CameraAdjustmentType::Azimuth,
                       this->Internal->azimuthAngle->value());
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::applyCameraZoomIn() {
    this->adjustCamera(CameraAdjustmentType::Zoom,
                       this->Internal->zoomFactor->value());
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::applyCameraZoomOut() {
    this->adjustCamera(CameraAdjustmentType::Zoom,
                       1.0 / this->Internal->zoomFactor->value());
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::ConfigureCustomViewpoints() {
    if (ecvCameraParamEditDlg::ConfigureCustomViewpoints(this)) {
        this->updateCustomViewpointButtons();
    }
}

//-----------------------------------------------------------------------------
bool ecvCameraParamEditDlg::ConfigureCustomViewpoints(QWidget* parentWidget) {
    QStringList toolTips = ecvCameraParamEditDlg::CustomViewpointToolTips();
    QStringList configs =
            ecvCameraParamEditDlg::CustomViewpointConfigurations();

    // user modifies the configuration
    QString currentConfig =
            ecvGenericCameraTool::CurrentCameraParam.toString().c_str();
    ecvCustomViewpointButtonDlg dialog(parentWidget, Qt::WindowFlags(),
                                       toolTips, configs, currentConfig);
    if (dialog.exec() == QDialog::Accepted) {
        // save the new configuration into the app wide settings.
        configs = dialog.getConfigurations();
        ecvSettingManager* settings = ecvSettingManager::TheInstance();
        settings->beginGroup("CustomViewButtons");
        settings->beginGroup("Configurations");
        settings->remove("");  // remove all items in the group.
        int index = 0;
        for (const QString& config : configs) {
            settings->setValue(QString::number(index++), config);
        }
        settings->endGroup();

        toolTips = dialog.getToolTips();
        settings->beginGroup("ToolTips");
        settings->remove("");  // remove all items in the group.
        index = 0;
        for (const QString& toolTip : toolTips) {
            settings->setValue(QString::number(index++), toolTip);
        }
        settings->endGroup();
        settings->endGroup();
        settings->alertSettingsModified();
        return true;
    }
    return false;
}

void ecvCameraParamEditDlg::addCurrentViewpointToCustomViewpoints() {
    if (AddCurrentViewpointToCustomViewpoints()) {
        this->updateCustomViewpointButtons();
    }
}

//-----------------------------------------------------------------------------
bool ecvCameraParamEditDlg::AddCurrentViewpointToCustomViewpoints() {
    // grab the current camera configuration.
    QString curCameraParam =
            ecvGenericCameraTool::CurrentCameraParam.toString().c_str();
    // load the existing button configurations from the app wide settings.
    QStringList configs =
            ecvCameraParamEditDlg::CustomViewpointConfigurations();

    // Add current viewpoint config to setting
    ecvSettingManager* settings = ecvSettingManager::TheInstance();
    settings->beginGroup("CustomViewButtons");
    settings->beginGroup("Configurations");
    settings->setValue(QString::number(configs.size()), curCameraParam);
    settings->endGroup();
    settings->beginGroup("ToolTips");
    settings->setValue(QString::number(configs.size()),
                       QString("Current Viewpoint %1").arg(configs.size() + 1));
    settings->endGroup();
    settings->endGroup();
    settings->alertSettingsModified();
    return true;
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::ApplyCustomViewpoint() {
    int buttonId = -1;
    if (QObject* asender = this->sender()) {
        buttonId = asender->property("pqCameraDialog_INDEX").toInt();
    } else {
        return;
    }

    if (ecvCameraParamEditDlg::ApplyCustomViewpoint(buttonId)) {
        // camera configuration has been modified update the scene.
        this->updateUi();
        this->reflectParamChange();
    }
}

//-----------------------------------------------------------------------------
bool ecvCameraParamEditDlg::ApplyCustomViewpoint(int CustomViewpointIndex) {
    ecvSettingManager* settings = ecvSettingManager::TheInstance();
    settings->beginGroup("CustomViewButtons");
    settings->beginGroup("Configurations");
    QString config = settings->value(QString::number(CustomViewpointIndex), "")
                             .toString();
    settings->endGroup();
    settings->endGroup();
    if (config.isEmpty()) {
        return false;
    }

    ecvGenericCameraTool::SaveBuffer();
    ecvGenericCameraTool::CurrentCameraParam.loadConfig(config);

    return true;
}

//-----------------------------------------------------------------------------
bool ecvCameraParamEditDlg::DeleteCustomViewpoint(int CustomViewpointIndex) {
    QStringList toolTips = ecvCameraParamEditDlg::CustomViewpointToolTips();
    if (CustomViewpointIndex >= toolTips.size()) {
        return false;
    }

    ecvSettingManager* settings = ecvSettingManager::TheInstance();
    settings->beginGroup("CustomViewButtons");
    settings->beginGroup("Configurations");
    for (int i = 0; i < toolTips.size() - 1; i++) {
        if (i < CustomViewpointIndex) {
            continue;
        }
        settings->setValue(QString::number(i),
                           settings->value(QString::number(i + 1)));
    }
    settings->remove(QString::number(toolTips.size() - 1));
    settings->endGroup();
    settings->beginGroup("ToolTips");
    for (int i = 0; i < toolTips.size() - 1; i++) {
        if (i < CustomViewpointIndex) {
            continue;
        }
        settings->setValue(QString::number(i),
                           settings->value(QString::number(i + 1)));
    }
    settings->remove(QString::number(toolTips.size() - 1));
    settings->endGroup();
    settings->endGroup();
    settings->alertSettingsModified();
    return true;
}

//-----------------------------------------------------------------------------
bool ecvCameraParamEditDlg::SetToCurrentViewpoint(int CustomViewpointIndex) {
    // Add current viewpoint config to setting
    ecvSettingManager* settings = ecvSettingManager::TheInstance();
    settings->beginGroup("CustomViewButtons");
    settings->beginGroup("Configurations");
    settings->setValue(
            QString::number(CustomViewpointIndex),
            QString(ecvGenericCameraTool::CurrentCameraParam.toString()
                            .c_str()));
    settings->endGroup();
    settings->endGroup();
    settings->alertSettingsModified();
    return true;
}

//-----------------------------------------------------------------------------
QStringList ecvCameraParamEditDlg::CustomViewpointConfigurations() {
    // Recover configurations from settings
    ecvSettingManager* settings = ecvSettingManager::TheInstance();
    settings->beginGroup("CustomViewButtons");
    settings->beginGroup("Configurations");
    const QStringList configs = getListOfStrings(
            settings, ecvCustomViewpointButtonDlg::DEFAULT_TOOLTIP,
            ecvCustomViewpointButtonDlg::MINIMUM_NUMBER_OF_ITEMS,
            ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS);
    settings->endGroup();
    settings->endGroup();
    return configs;
}

//-----------------------------------------------------------------------------
QStringList ecvCameraParamEditDlg::CustomViewpointToolTips() {
    // Recover tooltTips from settings
    ecvSettingManager* settings = ecvSettingManager::TheInstance();
    settings->beginGroup("CustomViewButtons");
    settings->beginGroup("ToolTips");
    const QStringList toolTips = getListOfStrings(
            settings, ecvCustomViewpointButtonDlg::DEFAULT_TOOLTIP,
            ecvCustomViewpointButtonDlg::MINIMUM_NUMBER_OF_ITEMS,
            ecvCustomViewpointButtonDlg::MAXIMUM_NUMBER_OF_ITEMS);
    settings->endGroup();
    settings->endGroup();
    return toolTips;
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::updateCustomViewpointButtons() {
    this->Internal->updateCustomViewpointButtons(this);
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::saveCameraConfiguration() {
    QString filters = ".cam";

    // default output path (+ filename)
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::SaveFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();

    // ask the user for the output filename
    QString selectedFilename = QFileDialog::getSaveFileName(
            this, tr("Save Custom Viewpoints Configuration"), currentPath,
            filters);

    if (selectedFilename.isEmpty()) {
        // process cancelled by the user
        return;
    }

    QString filename = selectedFilename;
    m_tool->saveCameraConfiguration(CVTools::FromQString(filename));
}

//-----------------------------------------------------------------------------
void ecvCameraParamEditDlg::loadCameraConfiguration() {
    QString filters = ".cam";

    // persistent settings
    QString currentPath =
            ecvSettingManager::getValue(ecvPS::LoadFile(), ecvPS::CurrentPath(),
                                        ecvFileUtils::defaultDocPath())
                    .toString();
    QStringList selectedFiles = QFileDialog::getOpenFileNames(
            this, QString("Load Custom Camera Configuration"), currentPath,
            filters);

    if (selectedFiles.isEmpty()) return;
    QString filename;
    filename = selectedFiles[0];
    m_tool->loadCameraConfiguration(CVTools::FromQString(filename));
}

void ecvCameraParamEditDlg::pickPointAsPivot(bool state) {
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
                    &ecvCameraParamEditDlg::processPickedItem);
        } else {
            ecvDisplayTools::SetPickingMode(ecvDisplayTools::DEFAULT_PICKING);
            disconnect(ecvDisplayTools::TheInstance(),
                       &ecvDisplayTools::itemPicked, this,
                       &ecvCameraParamEditDlg::processPickedItem);
        }
    }

    this->Internal->pivotPickingToolButton->blockSignals(true);
    this->Internal->pivotPickingToolButton->setChecked(state);
    this->Internal->pivotPickingToolButton->blockSignals(false);
}
