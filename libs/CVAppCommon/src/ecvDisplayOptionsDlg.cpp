// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvDisplayOptionsDlg.h"

#include "ui_displayOptionsDlg.h"

// local
#include "ecvApplicationBase.h"
#include "ecvPersistentSettings.h"
#include "ecvQtHelpers.h"
#include "ecvSettingManager.h"

// ECV_DB_LIB
#include <ecvColorTypes.h>
#include <ecvDisplayTools.h>

#include "ecvHObject.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QColor>
#include <QColorDialog>
#include <QMetaObject>
#include <QObject>
#include <QSettings>
#include <QStyleFactory>

// Standard
#include <algorithm>

// Default 'min cloud size' for LoD  when VBOs are activated
static const double s_defaultMaxVBOCloudSizeM = 50.0;

ccDisplayOptionsDlg::ccDisplayOptionsDlg(QWidget* parent)
    : QDialog(parent, Qt::Tool),
      m_ui(new Ui::DisplayOptionsDlg()),
      m_defaultAppStyleIndex(-1) {
    m_ui->setupUi(this);

    connect(m_ui->ambientColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeLightAmbientColor);
    connect(m_ui->diffuseColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeLightDiffuseColor);
    connect(m_ui->specularColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeLightSpecularColor);
    connect(m_ui->meshBackColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeMeshBackDiffuseColor);
    connect(m_ui->meshSpecularColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeMeshSpecularColor);
    connect(m_ui->meshFrontColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeMeshFrontDiffuseColor);
    connect(m_ui->bbColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeBBColor);
    connect(m_ui->showBBOnSelectedCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.showBBOnSelected = state; });
    connect(m_ui->bbOpacityDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccDisplayOptionsDlg::changeBBOpacity);
    connect(m_ui->bbLineWidthSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeBBLineWidth);
    connect(m_ui->bkgColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeBackgroundColor);
    connect(m_ui->labelBkgColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeLabelBackgroundColor);
    connect(m_ui->labelMarkerColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeLabelMarkerColor);
    connect(m_ui->pointsColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changePointsColor);
    connect(m_ui->textColorButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeTextColor);

    connect(m_ui->doubleSidedCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.lightDoubleSided = state; });
    connect(m_ui->enableGradientCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.drawBackgroundGradient = state; });
    connect(m_ui->showCrossCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.displayCross = state; });
    connect(m_ui->colorScaleShowHistogramCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.colorScaleShowHistogram = state; });
    connect(m_ui->useColorScaleShaderCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.colorScaleUseShader = state; });
    connect(m_ui->decimateMeshBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.decimateMeshOnMove = state; });
    connect(m_ui->decimateCloudBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.decimateCloudOnMove = state; });
    connect(m_ui->drawRoundedPointsCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { parameters.drawRoundedPoints = state; });
    connect(m_ui->autoDisplayNormalsCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { options.normalsDisplayedByDefault = state; });
    connect(m_ui->useNativeDialogsCheckBox, &QCheckBox::toggled, this,
            [&](bool state) { options.useNativeDialogs = state; });
    connect(m_ui->askForConfirmationBeforeQuittingCheckBox, &QCheckBox::toggled,
            this, [&](bool state) {
                options.askForConfirmationBeforeQuitting = state;
            });
    connect(m_ui->logVerbosityComboBox,
            static_cast<void (QComboBox::*)(int)>(
                    &QComboBox::currentIndexChanged),
            this, &ccDisplayOptionsDlg::changeLogVerbosityLevel);

    connect(m_ui->useVBOCheckBox, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::changeVBOUsage);

    connect(m_ui->colorRampWidthSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeColorScaleRampWidth);

    connect(m_ui->defaultFontSizeSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeDefaultFontSize);
    connect(m_ui->labelFontSizeSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeLabelFontSize);
    connect(m_ui->numberPrecisionSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeNumberPrecision);
    connect(m_ui->labelOpacitySpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeLabelOpacity);
    connect(m_ui->labelMarkerSizeSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccDisplayOptionsDlg::changeLabelMarkerSize);

    connect(m_ui->zoomSpeedDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccDisplayOptionsDlg::changeZoomSpeed);
    connect(m_ui->maxCloudSizeDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccDisplayOptionsDlg::changeMaxCloudSize);
    connect(m_ui->maxMeshSizeDoubleSpinBox,
            static_cast<void (QDoubleSpinBox::*)(double)>(
                    &QDoubleSpinBox::valueChanged),
            this, &ccDisplayOptionsDlg::changeMaxMeshSize);

    connect(m_ui->autoComputeOctreeComboBox,
            static_cast<void (QComboBox::*)(int)>(
                    &QComboBox::currentIndexChanged),
            this, &ccDisplayOptionsDlg::changeAutoComputeOctreeOption);

    connect(m_ui->appStyleComboBox,
            static_cast<void (QComboBox::*)(int)>(
                    &QComboBox::currentIndexChanged),
            this, &ccDisplayOptionsDlg::changeAppStyle);

    connect(m_ui->okButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::doAccept);
    connect(m_ui->applyButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::apply);
    connect(m_ui->resetButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::reset);
    connect(m_ui->cancelButton, &QAbstractButton::clicked, this,
            &ccDisplayOptionsDlg::doReject);

    // Populate application style combo box
    populateAppStyleComboBox();

    oldParameters = parameters = ecvGui::Parameters();
    oldOptions = options = ecvOptions::Instance();

    refresh();

    setUpdatesEnabled(true);
}

ccDisplayOptionsDlg::~ccDisplayOptionsDlg() {
    delete m_ui;
    m_ui = nullptr;
}

void ccDisplayOptionsDlg::refresh() {
    const ecvColor::Rgbaf& ac = parameters.lightAmbientColor;
    lightAmbientColor.setRgbF(ac.r, ac.g, ac.b, ac.a);
    ccQtHelpers::SetButtonColor(m_ui->ambientColorButton, lightAmbientColor);

    const ecvColor::Rgbaf& dc = parameters.lightDiffuseColor;
    lightDiffuseColor.setRgbF(dc.r, dc.g, dc.b, dc.a);
    ccQtHelpers::SetButtonColor(m_ui->diffuseColorButton, lightDiffuseColor);

    const ecvColor::Rgbaf& sc = parameters.lightSpecularColor;
    lightSpecularColor.setRgbF(sc.r, sc.g, sc.b, sc.a);
    ccQtHelpers::SetButtonColor(m_ui->specularColorButton, lightSpecularColor);

    const ecvColor::Rgbaf& mbc = parameters.meshBackDiff;
    meshBackDiff.setRgbF(mbc.r, mbc.g, mbc.b, mbc.a);
    ccQtHelpers::SetButtonColor(m_ui->meshBackColorButton, meshBackDiff);

    const ecvColor::Rgbaf& mspec = parameters.meshSpecular;
    meshSpecularColor.setRgbF(mspec.r, mspec.g, mspec.b, mspec.a);
    ccQtHelpers::SetButtonColor(m_ui->meshSpecularColorButton,
                                meshSpecularColor);

    const ecvColor::Rgbaf& mfc = parameters.meshFrontDiff;
    meshFrontDiff.setRgbF(mfc.r, mfc.g, mfc.b, mfc.a);
    ccQtHelpers::SetButtonColor(m_ui->meshFrontColorButton, meshFrontDiff);

    const ecvColor::Rgbub& bbc = parameters.bbDefaultCol;
    bbDefaultCol.setRgb(bbc.r, bbc.g, bbc.b);
    ccQtHelpers::SetButtonColor(m_ui->bbColorButton, bbDefaultCol);
    m_ui->showBBOnSelectedCheckBox->setChecked(parameters.showBBOnSelected);
    m_ui->bbOpacityDoubleSpinBox->setValue(parameters.bbOpacity);
    m_ui->bbLineWidthSpinBox->setValue(parameters.bbLineWidth);

    const ecvColor::Rgbub& bgc = parameters.backgroundCol;
    backgroundCol.setRgb(bgc.r, bgc.g, bgc.b);
    ccQtHelpers::SetButtonColor(m_ui->bkgColorButton, backgroundCol);

    const ecvColor::Rgbub& lblbc = parameters.labelBackgroundCol;
    labelBackgroundCol.setRgb(lblbc.r, lblbc.g, lblbc.b);
    ccQtHelpers::SetButtonColor(m_ui->labelBkgColorButton, labelBackgroundCol);

    const ecvColor::Rgbub& lblmc = parameters.labelMarkerCol;
    labelMarkerCol.setRgb(lblmc.r, lblmc.g, lblmc.b);
    ccQtHelpers::SetButtonColor(m_ui->labelMarkerColorButton, labelMarkerCol);

    const ecvColor::Rgbub& pdc = parameters.pointsDefaultCol;
    pointsDefaultCol.setRgb(pdc.r, pdc.g, pdc.b);
    ccQtHelpers::SetButtonColor(m_ui->pointsColorButton, pointsDefaultCol);

    const ecvColor::Rgbub& tdc = parameters.textDefaultCol;
    textDefaultCol.setRgb(tdc.r, tdc.g, tdc.b);
    ccQtHelpers::SetButtonColor(m_ui->textColorButton, textDefaultCol);

    m_ui->doubleSidedCheckBox->setChecked(parameters.lightDoubleSided);
    m_ui->enableGradientCheckBox->setChecked(parameters.drawBackgroundGradient);
    m_ui->decimateMeshBox->setChecked(parameters.decimateMeshOnMove);
    m_ui->maxMeshSizeDoubleSpinBox->setValue(parameters.minLoDMeshSize /
                                             1000000.0);
    m_ui->decimateCloudBox->setChecked(parameters.decimateCloudOnMove);
    m_ui->drawRoundedPointsCheckBox->setChecked(parameters.drawRoundedPoints);
    m_ui->maxCloudSizeDoubleSpinBox->setValue(parameters.minLoDCloudSize /
                                              1000000.0);
    m_ui->useVBOCheckBox->setChecked(parameters.useVBOs);
    m_ui->showCrossCheckBox->setChecked(parameters.displayCross);

    m_ui->colorScaleShowHistogramCheckBox->setChecked(
            parameters.colorScaleShowHistogram);
    m_ui->useColorScaleShaderCheckBox->setChecked(
            parameters.colorScaleUseShader);
    m_ui->useColorScaleShaderCheckBox->setEnabled(
            parameters.colorScaleShaderSupported);
    m_ui->colorRampWidthSpinBox->setValue(parameters.colorScaleRampWidth);

    m_ui->defaultFontSizeSpinBox->setValue(parameters.defaultFontSize);
    m_ui->labelFontSizeSpinBox->setValue(parameters.labelFontSize);
    m_ui->numberPrecisionSpinBox->setValue(parameters.displayedNumPrecision);
    m_ui->labelOpacitySpinBox->setValue(parameters.labelOpacity);
    m_ui->labelMarkerSizeSpinBox->setValue(parameters.labelMarkerSize);

    m_ui->zoomSpeedDoubleSpinBox->setValue(parameters.zoomSpeed);

    m_ui->autoComputeOctreeComboBox->setCurrentIndex(
            parameters.autoComputeOctree);

    m_ui->autoDisplayNormalsCheckBox->setChecked(
            options.normalsDisplayedByDefault);
    m_ui->useNativeDialogsCheckBox->setChecked(options.useNativeDialogs);
    m_ui->askForConfirmationBeforeQuittingCheckBox->setChecked(
            options.askForConfirmationBeforeQuitting);
    m_ui->logVerbosityComboBox->setCurrentIndex(
            std::min(static_cast<int>(options.logVerbosityLevel),
                     static_cast<int>(CVLog::LOG_WARNING)));

    update();
}

void ccDisplayOptionsDlg::changeLightDiffuseColor() {
    QColor newCol = QColorDialog::getColor(lightDiffuseColor, this);
    if (!newCol.isValid()) return;

    lightDiffuseColor = newCol;
    ccQtHelpers::SetButtonColor(m_ui->diffuseColorButton, lightDiffuseColor);
    parameters.lightDiffuseColor = ecvColor::FromQColoraf(lightDiffuseColor);
}

void ccDisplayOptionsDlg::changeLightAmbientColor() {
    QColor newCol = QColorDialog::getColor(lightAmbientColor, this);
    if (!newCol.isValid()) return;

    lightAmbientColor = newCol;
    ccQtHelpers::SetButtonColor(m_ui->ambientColorButton, lightAmbientColor);
    parameters.lightAmbientColor = ecvColor::FromQColoraf(lightAmbientColor);

    update();
}

void ccDisplayOptionsDlg::changeLightSpecularColor() {
    QColor newCol = QColorDialog::getColor(lightSpecularColor, this);
    if (!newCol.isValid()) return;

    lightSpecularColor = newCol;
    ccQtHelpers::SetButtonColor(m_ui->specularColorButton, lightSpecularColor);
    parameters.lightSpecularColor = ecvColor::FromQColoraf(lightSpecularColor);

    update();
}

void ccDisplayOptionsDlg::changeMeshFrontDiffuseColor() {
    QColor newCol = QColorDialog::getColor(meshFrontDiff, this);
    if (!newCol.isValid()) return;

    meshFrontDiff = newCol;
    ccQtHelpers::SetButtonColor(m_ui->meshFrontColorButton, meshFrontDiff);

    parameters.meshFrontDiff = ecvColor::FromQColoraf(meshFrontDiff);

    update();
}

void ccDisplayOptionsDlg::changeMeshBackDiffuseColor() {
    QColor newCol = QColorDialog::getColor(meshBackDiff, this);
    if (!newCol.isValid()) return;

    meshBackDiff = newCol;
    ccQtHelpers::SetButtonColor(m_ui->meshBackColorButton, meshBackDiff);
    parameters.meshBackDiff = ecvColor::FromQColoraf(meshBackDiff);

    update();
}

void ccDisplayOptionsDlg::changeMeshSpecularColor() {
    QColor newCol = QColorDialog::getColor(meshSpecularColor, this);
    if (!newCol.isValid()) return;

    meshSpecularColor = newCol;
    ccQtHelpers::SetButtonColor(m_ui->meshSpecularColorButton,
                                meshSpecularColor);
    parameters.meshSpecular = ecvColor::FromQColoraf(meshSpecularColor);

    update();
}

void ccDisplayOptionsDlg::changePointsColor() {
    QColor newCol = QColorDialog::getColor(pointsDefaultCol, this);
    if (!newCol.isValid()) return;

    pointsDefaultCol = newCol;
    ccQtHelpers::SetButtonColor(m_ui->pointsColorButton, pointsDefaultCol);
    parameters.pointsDefaultCol = ecvColor::FromQColor(pointsDefaultCol);

    update();
}

void ccDisplayOptionsDlg::changeBBColor() {
    QColor newCol = QColorDialog::getColor(bbDefaultCol, this);
    if (!newCol.isValid()) return;

    bbDefaultCol = newCol;
    ccQtHelpers::SetButtonColor(m_ui->bbColorButton, bbDefaultCol);
    parameters.bbDefaultCol = ecvColor::FromQColor(bbDefaultCol);

    update();
}

void ccDisplayOptionsDlg::changeTextColor() {
    QColor newCol = QColorDialog::getColor(textDefaultCol, this);
    if (!newCol.isValid()) return;

    textDefaultCol = newCol;
    ccQtHelpers::SetButtonColor(m_ui->textColorButton, textDefaultCol);
    parameters.textDefaultCol = ecvColor::FromQColor(textDefaultCol);

    update();
}

void ccDisplayOptionsDlg::changeBackgroundColor() {
    QColor newCol = QColorDialog::getColor(backgroundCol, this);
    if (!newCol.isValid()) return;

    backgroundCol = newCol;
    ccQtHelpers::SetButtonColor(m_ui->bkgColorButton, backgroundCol);
    parameters.backgroundCol = ecvColor::FromQColor(backgroundCol);

    update();
}

void ccDisplayOptionsDlg::changeLabelBackgroundColor() {
    QColor newCol = QColorDialog::getColor(labelBackgroundCol, this);
    if (!newCol.isValid()) return;

    labelBackgroundCol = newCol;
    ccQtHelpers::SetButtonColor(m_ui->labelBkgColorButton, labelBackgroundCol);
    parameters.labelBackgroundCol = ecvColor::FromQColor(labelBackgroundCol);

    update();
}

void ccDisplayOptionsDlg::changeLabelMarkerColor() {
    QColor newCol = QColorDialog::getColor(labelMarkerCol, this);
    if (!newCol.isValid()) return;

    labelMarkerCol = newCol;
    ccQtHelpers::SetButtonColor(m_ui->labelMarkerColorButton, labelMarkerCol);

    parameters.labelMarkerCol = ecvColor::FromQColor(labelMarkerCol);

    update();
}

void ccDisplayOptionsDlg::changeMaxMeshSize(double val) {
    parameters.minLoDMeshSize = static_cast<unsigned>(val * 1000000);
}

void ccDisplayOptionsDlg::changeMaxCloudSize(double val) {
    parameters.minLoDCloudSize = static_cast<unsigned>(val * 1000000);
}

void ccDisplayOptionsDlg::changeVBOUsage() {
    parameters.useVBOs = m_ui->useVBOCheckBox->isChecked();
    if (parameters.useVBOs &&
        m_ui->maxCloudSizeDoubleSpinBox->value() < s_defaultMaxVBOCloudSizeM) {
        m_ui->maxCloudSizeDoubleSpinBox->setValue(s_defaultMaxVBOCloudSizeM);
    }
}

void ccDisplayOptionsDlg::changeColorScaleRampWidth(int val) {
    if (val < 2) return;
    parameters.colorScaleRampWidth = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::changeDefaultFontSize(int val) {
    if (val < 0) return;
    parameters.defaultFontSize = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::changeLabelFontSize(int val) {
    if (val < 0) return;
    parameters.labelFontSize = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::changeNumberPrecision(int val) {
    if (val < 0) return;
    parameters.displayedNumPrecision = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::changeZoomSpeed(double val) {
    parameters.zoomSpeed = val;
}

void ccDisplayOptionsDlg::changeAutoComputeOctreeOption(int index) {
    assert(index >= 0 && index < 3);
    parameters.autoComputeOctree =
            static_cast<ecvGui::ParamStruct::ComputeOctreeForPicking>(index);
}

void ccDisplayOptionsDlg::changeLabelOpacity(int val) {
    if (val < 0 || val > 100) return;
    parameters.labelOpacity = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::changeLabelMarkerSize(int val) {
    if (val <= 0) return;

    parameters.labelMarkerSize = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::changeBBOpacity(double val) {
    if (val < 0.0 || val > 1.0) return;
    parameters.bbOpacity = val;
}

void ccDisplayOptionsDlg::changeBBLineWidth(int val) {
    if (val < 1) return;
    parameters.bbLineWidth = static_cast<unsigned>(val);
}

void ccDisplayOptionsDlg::doReject() {
    // Restore old parameters and options
    ecvGui::Set(oldParameters);
    ecvOptions::Set(oldOptions);

    // Restore old application style
    if (m_defaultAppStyleIndex >= 0) {
        QString oldStyle =
                m_ui->appStyleComboBox->itemText(m_defaultAppStyleIndex);
        if (ecvApp) {
            ecvApp->setAppStyle(oldStyle);
        }
    }

    // Force redraw of selected objects to restore BoundingBox properties
    ccHObject* sceneDB = ecvDisplayTools::GetSceneDB();
    if (sceneDB) {
        // Find all selected entities and force them to redraw
        ccHObject::Container allEntities;
        sceneDB->filterChildren(allEntities, true, CV_TYPES::OBJECT);
        for (ccHObject* entity : allEntities) {
            if (entity && entity->isSelected()) {
                entity->setForceRedrawRecursive(true);
            }
        }
    }

    emit aspectHasChanged();

    reject();
}

void ccDisplayOptionsDlg::reset() {
    parameters.reset();
    options.reset();

    // Reset app style to default
    if (m_defaultAppStyleIndex >= 0) {
        m_ui->appStyleComboBox->setCurrentIndex(m_defaultAppStyleIndex);
    }

    refresh();
}

void ccDisplayOptionsDlg::apply() {
    ecvGui::Set(parameters);
    ecvOptions::Set(options);

    // Apply application style
    {
        QString style = m_ui->appStyleComboBox->currentText();
        ecvApp->setAppStyle(style);
    }

    // Apply log verbosity level (now directly uses CVLog::MessageLevelFlags)
    {
        if (CVLog::VerbosityLevel() != options.logVerbosityLevel) {
            CVLog::SetVerbosityLevel(options.logVerbosityLevel);
            CVLog::Print(QString("New log verbosity level: %1")
                                 .arg(options.logVerbosityLevel));
        }
    }

    // Force redraw of selected objects to update BoundingBox properties
    ccHObject* sceneDB = ecvDisplayTools::GetSceneDB();
    if (sceneDB) {
        // Find all selected entities and force them to redraw
        ccHObject::Container allEntities;
        sceneDB->filterChildren(allEntities, true, CV_TYPES::OBJECT);
        for (ccHObject* entity : allEntities) {
            if (entity && entity->isSelected()) {
                entity->setForceRedrawRecursive(true);
            }
        }
    }

    emit aspectHasChanged();
}

void ccDisplayOptionsDlg::changeAppStyle(int index) {
    // Optional: could add live preview here
    Q_UNUSED(index);
}

void ccDisplayOptionsDlg::changeLogVerbosityLevel(int index) {
    if (index >= 0 && index < CVLog::LOG_ERROR) {
        options.logVerbosityLevel =
                static_cast<CVLog::MessageLevelFlags>(index);
    } else {
        // unexpected value
        assert(false);
    }
}

void ccDisplayOptionsDlg::populateAppStyleComboBox() {
    // Get currently active style
    // Get the current/default style from settings
    // (matching CloudCompare's approach using QSettings)
    QSettings settings;
    settings.beginGroup(ecvPS::AppStyle());
    QString defaultStyleName = settings.value("style").toString();
    settings.endGroup();

    // Fill with all available Qt styles
    QStringList appStyles = QStyleFactory::keys();
    for (const QString& style : appStyles) {
        m_ui->appStyleComboBox->addItem(style);
    }

    // Add custom dark/light themes from QDarkStyleSheet
    m_ui->appStyleComboBox->addItem(QStringLiteral("QDarkStyleSheet::Light"));
    m_ui->appStyleComboBox->addItem(QStringLiteral("QDarkStyleSheet::Dark"));

    // Find and set the current style (matching CloudCompare's logic)
    // Handle case-insensitive comparison and macOS style name aliases
    for (int i = 0; i < m_ui->appStyleComboBox->count(); ++i) {
        QString itemText = m_ui->appStyleComboBox->itemText(i);
        if (itemText.compare(defaultStyleName, Qt::CaseInsensitive) == 0) {
            m_defaultAppStyleIndex = i;
            break;
        }
    }

    // On macOS, handle style name aliases (macOS <-> macintosh)
#ifdef Q_OS_MAC
    if (m_defaultAppStyleIndex < 0 && !defaultStyleName.isEmpty()) {
        // Try to match macOS/macintosh style names (they are equivalent)
        if (defaultStyleName.compare("macOS", Qt::CaseInsensitive) == 0 ||
            defaultStyleName.compare("macintosh", Qt::CaseInsensitive) == 0) {
            for (int i = 0; i < m_ui->appStyleComboBox->count(); ++i) {
                QString itemText = m_ui->appStyleComboBox->itemText(i);
                if (itemText.compare("macOS", Qt::CaseInsensitive) == 0 ||
                    itemText.compare("macintosh", Qt::CaseInsensitive) == 0) {
                    m_defaultAppStyleIndex = i;
                    break;
                }
            }
        }
    }
#endif

    // Set default index (use 0 if no match found)
    if (m_defaultAppStyleIndex < 0) {
        m_defaultAppStyleIndex = 0;
    }
    m_ui->appStyleComboBox->setCurrentIndex(m_defaultAppStyleIndex);
}

void ccDisplayOptionsDlg::doAccept() {
    apply();

    parameters.toPersistentSettings();
    options.toPersistentSettings();

    accept();
}
