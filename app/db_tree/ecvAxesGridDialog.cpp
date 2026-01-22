// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAxesGridDialog.h"

#include <QCheckBox>
#include <QColorDialog>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include "ecvCustomLabelsWidget.h"

ecvAxesGridDialog::ecvAxesGridDialog(const QString& title, QWidget* parent)
    : QDialog(parent), m_currentColor(127, 127, 127) {
    setWindowTitle(title);
    // Non-modal for real-time preview (ParaView-style)
    setWindowModality(Qt::NonModal);
    setWindowFlags(Qt::Window | Qt::WindowTitleHint |
                   Qt::WindowCloseButtonHint | Qt::WindowStaysOnTopHint);
    setupUI();
    setMinimumWidth(400);
    setMinimumHeight(500);
}

ecvAxesGridDialog::~ecvAxesGridDialog() = default;

void ecvAxesGridDialog::setupUI() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(8);
    mainLayout->setContentsMargins(10, 10, 10, 10);

    // Section 1: Title Texts
    QGroupBox* titleTextsGroup = new QGroupBox(tr("Title Texts"), this);
    QFormLayout* titleTextsLayout = new QFormLayout(titleTextsGroup);
    titleTextsLayout->setContentsMargins(8, 12, 8, 8);

    m_xTitleEdit = new QLineEdit("X Axis", this);
    titleTextsLayout->addRow(tr("X Title:"), m_xTitleEdit);

    m_yTitleEdit = new QLineEdit("Y Axis", this);
    titleTextsLayout->addRow(tr("Y Title:"), m_yTitleEdit);

    m_zTitleEdit = new QLineEdit("Z Axis", this);
    titleTextsLayout->addRow(tr("Z Title:"), m_zTitleEdit);

    mainLayout->addWidget(titleTextsGroup);

    // Section 2: Face Properties
    QGroupBox* facePropertiesGroup = new QGroupBox(tr("Face Properties"), this);
    QFormLayout* faceLayout = new QFormLayout(facePropertiesGroup);
    faceLayout->setContentsMargins(8, 12, 8, 8);

    m_gridColorButton = new QPushButton(this);
    m_gridColorButton->setFixedSize(80, 24);
    updateColorButton();
    connect(m_gridColorButton, &QPushButton::clicked, this,
            &ecvAxesGridDialog::onColorButtonClicked);
    faceLayout->addRow(tr("Grid Color:"), m_gridColorButton);

    m_showGridCheckBox = new QCheckBox(this);
    m_showGridCheckBox->setChecked(false);
    faceLayout->addRow(tr("Show Grid:"), m_showGridCheckBox);

    mainLayout->addWidget(facePropertiesGroup);

    // Section 3: X Axis
    m_xAxisGroup = new QGroupBox(tr("X Axis Label Properties"), this);
    QVBoxLayout* xAxisLayout = new QVBoxLayout(m_xAxisGroup);
    xAxisLayout->setContentsMargins(8, 12, 8, 8);

    m_xAxisUseCustomLabelsCheckBox =
            new QCheckBox(tr("X Axis Use Custom Labels"), this);
    xAxisLayout->addWidget(m_xAxisUseCustomLabelsCheckBox);

    m_xAxisCustomLabelsWidget = new ecvCustomLabelsWidget(this);
    m_xAxisCustomLabelsWidget->setVisible(false);
    xAxisLayout->addWidget(m_xAxisCustomLabelsWidget);

    connect(m_xAxisUseCustomLabelsCheckBox, &QCheckBox::toggled, this,
            &ecvAxesGridDialog::onXAxisUseCustomLabelsToggled);

    mainLayout->addWidget(m_xAxisGroup);

    // Section 4: Y Axis
    m_yAxisGroup = new QGroupBox(tr("Y Axis Label Properties"), this);
    QVBoxLayout* yAxisLayout = new QVBoxLayout(m_yAxisGroup);
    yAxisLayout->setContentsMargins(8, 12, 8, 8);

    m_yAxisUseCustomLabelsCheckBox =
            new QCheckBox(tr("Y Axis Use Custom Labels"), this);
    yAxisLayout->addWidget(m_yAxisUseCustomLabelsCheckBox);

    m_yAxisCustomLabelsWidget = new ecvCustomLabelsWidget(this);
    m_yAxisCustomLabelsWidget->setVisible(false);
    yAxisLayout->addWidget(m_yAxisCustomLabelsWidget);

    connect(m_yAxisUseCustomLabelsCheckBox, &QCheckBox::toggled, this,
            &ecvAxesGridDialog::onYAxisUseCustomLabelsToggled);

    mainLayout->addWidget(m_yAxisGroup);

    // Section 5: Z Axis
    m_zAxisGroup = new QGroupBox(tr("Z Axis Label Properties"), this);
    QVBoxLayout* zAxisLayout = new QVBoxLayout(m_zAxisGroup);
    zAxisLayout->setContentsMargins(8, 12, 8, 8);

    m_zAxisUseCustomLabelsCheckBox =
            new QCheckBox(tr("Z Axis Use Custom Labels"), this);
    zAxisLayout->addWidget(m_zAxisUseCustomLabelsCheckBox);

    m_zAxisCustomLabelsWidget = new ecvCustomLabelsWidget(this);
    m_zAxisCustomLabelsWidget->setVisible(false);
    zAxisLayout->addWidget(m_zAxisCustomLabelsWidget);

    connect(m_zAxisUseCustomLabelsCheckBox, &QCheckBox::toggled, this,
            &ecvAxesGridDialog::onZAxisUseCustomLabelsToggled);

    mainLayout->addWidget(m_zAxisGroup);

    // Section 6: Bounds (ParaView-style)
    QGroupBox* boundsGroup = new QGroupBox(tr("Bounds"), this);
    QVBoxLayout* boundsLayout = new QVBoxLayout(boundsGroup);
    boundsLayout->setContentsMargins(8, 12, 8, 8);

    m_useCustomBoundsCheckBox = new QCheckBox(tr("Use Custom Bounds"), this);
    boundsLayout->addWidget(m_useCustomBoundsCheckBox);

    // Custom bounds input fields (6 spinboxes: xmin, xmax, ymin, ymax, zmin,
    // zmax)
    m_customBoundsWidget = new QWidget(this);
    QFormLayout* boundsInputLayout = new QFormLayout(m_customBoundsWidget);
    boundsInputLayout->setContentsMargins(20, 8, 8,
                                          8);  // Indent for visual hierarchy

    m_xMinSpinBox = new QDoubleSpinBox(m_customBoundsWidget);
    m_xMinSpinBox->setRange(-1e10, 1e10);
    m_xMinSpinBox->setDecimals(3);
    m_xMinSpinBox->setValue(0.0);  // Default: 0.0
    boundsInputLayout->addRow(tr("X Min:"), m_xMinSpinBox);

    m_xMaxSpinBox = new QDoubleSpinBox(m_customBoundsWidget);
    m_xMaxSpinBox->setRange(-1e10, 1e10);
    m_xMaxSpinBox->setDecimals(3);
    m_xMaxSpinBox->setValue(1.0);  // Default: 1.0
    boundsInputLayout->addRow(tr("X Max:"), m_xMaxSpinBox);

    m_yMinSpinBox = new QDoubleSpinBox(m_customBoundsWidget);
    m_yMinSpinBox->setRange(-1e10, 1e10);
    m_yMinSpinBox->setDecimals(3);
    m_yMinSpinBox->setValue(0.0);  // Default: 0.0
    boundsInputLayout->addRow(tr("Y Min:"), m_yMinSpinBox);

    m_yMaxSpinBox = new QDoubleSpinBox(m_customBoundsWidget);
    m_yMaxSpinBox->setRange(-1e10, 1e10);
    m_yMaxSpinBox->setDecimals(3);
    m_yMaxSpinBox->setValue(1.0);  // Default: 1.0
    boundsInputLayout->addRow(tr("Y Max:"), m_yMaxSpinBox);

    m_zMinSpinBox = new QDoubleSpinBox(m_customBoundsWidget);
    m_zMinSpinBox->setRange(-1e10, 1e10);
    m_zMinSpinBox->setDecimals(3);
    m_zMinSpinBox->setValue(0.0);  // Default: 0.0
    boundsInputLayout->addRow(tr("Z Min:"), m_zMinSpinBox);

    m_zMaxSpinBox = new QDoubleSpinBox(m_customBoundsWidget);
    m_zMaxSpinBox->setRange(-1e10, 1e10);
    m_zMaxSpinBox->setDecimals(3);
    m_zMaxSpinBox->setValue(1.0);  // Default: 1.0
    boundsInputLayout->addRow(tr("Z Max:"), m_zMaxSpinBox);

    m_customBoundsWidget->setVisible(
            false);  // Initially hidden (ParaView default)
    boundsLayout->addWidget(m_customBoundsWidget);

    connect(m_useCustomBoundsCheckBox, &QCheckBox::toggled, this,
            &ecvAxesGridDialog::onUseCustomBoundsToggled);

    mainLayout->addWidget(boundsGroup);

    // Buttons
    QDialogButtonBox* buttonBox = new QDialogButtonBox(this);
    QPushButton* applyButton = buttonBox->addButton(QDialogButtonBox::Apply);
    QPushButton* resetButton = buttonBox->addButton(QDialogButtonBox::Reset);
    buttonBox->addButton(QDialogButtonBox::Cancel);
    buttonBox->addButton(QDialogButtonBox::Ok);

    connect(applyButton, &QPushButton::clicked, this,
            &ecvAxesGridDialog::onApply);
    connect(resetButton, &QPushButton::clicked, this,
            &ecvAxesGridDialog::onReset);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    mainLayout->addWidget(buttonBox);

    storeInitialValues();
}

void ecvAxesGridDialog::updateColorButton() {
    QString styleSheet = QString("QPushButton { background-color: rgb(%1, %2, "
                                 "%3); border: 1px solid #999; }")
                                 .arg(m_currentColor.red())
                                 .arg(m_currentColor.green())
                                 .arg(m_currentColor.blue());
    m_gridColorButton->setStyleSheet(styleSheet);
}

void ecvAxesGridDialog::onColorButtonClicked() {
    QColor newColor = QColorDialog::getColor(m_currentColor, this,
                                             tr("Select Grid Color"));
    if (newColor.isValid()) {
        m_currentColor = newColor;
        updateColorButton();
    }
}

void ecvAxesGridDialog::storeInitialValues() {
    m_initialXTitle = m_xTitleEdit->text();
    m_initialYTitle = m_yTitleEdit->text();
    m_initialZTitle = m_zTitleEdit->text();
    m_initialColor = m_currentColor;
    m_initialShowGrid = m_showGridCheckBox->isChecked();
    m_initialXCustomLabels = m_xAxisUseCustomLabelsCheckBox->isChecked();
    m_initialYCustomLabels = m_yAxisUseCustomLabelsCheckBox->isChecked();
    m_initialZCustomLabels = m_zAxisUseCustomLabelsCheckBox->isChecked();
    m_initialXLabels = m_xAxisCustomLabelsWidget->getLabels();
    m_initialYLabels = m_yAxisCustomLabelsWidget->getLabels();
    m_initialZLabels = m_zAxisCustomLabelsWidget->getLabels();
    m_initialCustomBounds = m_useCustomBoundsCheckBox->isChecked();

    // Store custom bounds values
    m_initialXMin = m_xMinSpinBox->value();
    m_initialXMax = m_xMaxSpinBox->value();
    m_initialYMin = m_yMinSpinBox->value();
    m_initialYMax = m_yMaxSpinBox->value();
    m_initialZMin = m_zMinSpinBox->value();
    m_initialZMax = m_zMaxSpinBox->value();
}

void ecvAxesGridDialog::restoreInitialValues() {
    m_xTitleEdit->setText(m_initialXTitle);
    m_yTitleEdit->setText(m_initialYTitle);
    m_zTitleEdit->setText(m_initialZTitle);
    m_currentColor = m_initialColor;
    updateColorButton();
    m_showGridCheckBox->setChecked(m_initialShowGrid);
    m_xAxisUseCustomLabelsCheckBox->setChecked(m_initialXCustomLabels);
    m_yAxisUseCustomLabelsCheckBox->setChecked(m_initialYCustomLabels);
    m_zAxisUseCustomLabelsCheckBox->setChecked(m_initialZCustomLabels);
    m_xAxisCustomLabelsWidget->setLabels(m_initialXLabels);
    m_yAxisCustomLabelsWidget->setLabels(m_initialYLabels);
    m_zAxisCustomLabelsWidget->setLabels(m_initialZLabels);
    m_useCustomBoundsCheckBox->setChecked(m_initialCustomBounds);

    // Restore custom bounds values
    m_xMinSpinBox->setValue(m_initialXMin);
    m_xMaxSpinBox->setValue(m_initialXMax);
    m_yMinSpinBox->setValue(m_initialYMin);
    m_yMaxSpinBox->setValue(m_initialYMax);
    m_zMinSpinBox->setValue(m_initialZMin);
    m_zMaxSpinBox->setValue(m_initialZMax);
}

void ecvAxesGridDialog::onApply() {
    emit applyRequested();
    emit propertiesChanged();
}

void ecvAxesGridDialog::onReset() {
    restoreInitialValues();
    emit applyRequested();
    emit propertiesChanged();
}

void ecvAxesGridDialog::onXAxisUseCustomLabelsToggled(bool checked) {
    m_xAxisCustomLabelsWidget->setVisible(checked);
    adjustSize();
}

void ecvAxesGridDialog::onYAxisUseCustomLabelsToggled(bool checked) {
    m_yAxisCustomLabelsWidget->setVisible(checked);
    adjustSize();
}

void ecvAxesGridDialog::onZAxisUseCustomLabelsToggled(bool checked) {
    m_zAxisCustomLabelsWidget->setVisible(checked);
    adjustSize();
}

// Getters and Setters
QString ecvAxesGridDialog::getXTitle() const { return m_xTitleEdit->text(); }
void ecvAxesGridDialog::setXTitle(const QString& title) {
    m_xTitleEdit->setText(title);
}
QString ecvAxesGridDialog::getYTitle() const { return m_yTitleEdit->text(); }
void ecvAxesGridDialog::setYTitle(const QString& title) {
    m_yTitleEdit->setText(title);
}
QString ecvAxesGridDialog::getZTitle() const { return m_zTitleEdit->text(); }
void ecvAxesGridDialog::setZTitle(const QString& title) {
    m_zTitleEdit->setText(title);
}

QColor ecvAxesGridDialog::getGridColor() const { return m_currentColor; }
void ecvAxesGridDialog::setGridColor(const QColor& color) {
    m_currentColor = color;
    updateColorButton();
}
bool ecvAxesGridDialog::getShowGrid() const {
    return m_showGridCheckBox->isChecked();
}
void ecvAxesGridDialog::setShowGrid(bool show) {
    m_showGridCheckBox->setChecked(show);
}

bool ecvAxesGridDialog::getXAxisUseCustomLabels() const {
    return m_xAxisUseCustomLabelsCheckBox->isChecked();
}
void ecvAxesGridDialog::setXAxisUseCustomLabels(bool use) {
    m_xAxisUseCustomLabelsCheckBox->setChecked(use);
}
QList<QPair<double, QString>> ecvAxesGridDialog::getXAxisCustomLabels() const {
    return m_xAxisCustomLabelsWidget->getLabels();
}
void ecvAxesGridDialog::setXAxisCustomLabels(
        const QList<QPair<double, QString>>& labels) {
    m_xAxisCustomLabelsWidget->setLabels(labels);
}

bool ecvAxesGridDialog::getYAxisUseCustomLabels() const {
    return m_yAxisUseCustomLabelsCheckBox->isChecked();
}
void ecvAxesGridDialog::setYAxisUseCustomLabels(bool use) {
    m_yAxisUseCustomLabelsCheckBox->setChecked(use);
}
QList<QPair<double, QString>> ecvAxesGridDialog::getYAxisCustomLabels() const {
    return m_yAxisCustomLabelsWidget->getLabels();
}
void ecvAxesGridDialog::setYAxisCustomLabels(
        const QList<QPair<double, QString>>& labels) {
    m_yAxisCustomLabelsWidget->setLabels(labels);
}

bool ecvAxesGridDialog::getZAxisUseCustomLabels() const {
    return m_zAxisUseCustomLabelsCheckBox->isChecked();
}
void ecvAxesGridDialog::setZAxisUseCustomLabels(bool use) {
    m_zAxisUseCustomLabelsCheckBox->setChecked(use);
}
QList<QPair<double, QString>> ecvAxesGridDialog::getZAxisCustomLabels() const {
    return m_zAxisCustomLabelsWidget->getLabels();
}
void ecvAxesGridDialog::setZAxisCustomLabels(
        const QList<QPair<double, QString>>& labels) {
    m_zAxisCustomLabelsWidget->setLabels(labels);
}

bool ecvAxesGridDialog::getUseCustomBounds() const {
    return m_useCustomBoundsCheckBox->isChecked();
}
void ecvAxesGridDialog::setUseCustomBounds(bool use) {
    m_useCustomBoundsCheckBox->setChecked(use);
}

void ecvAxesGridDialog::onUseCustomBoundsToggled(bool checked) {
    m_customBoundsWidget->setVisible(checked);
    adjustSize();
}

double ecvAxesGridDialog::getXMin() const { return m_xMinSpinBox->value(); }
void ecvAxesGridDialog::setXMin(double value) {
    m_xMinSpinBox->setValue(value);
}
double ecvAxesGridDialog::getXMax() const { return m_xMaxSpinBox->value(); }
void ecvAxesGridDialog::setXMax(double value) {
    m_xMaxSpinBox->setValue(value);
}
double ecvAxesGridDialog::getYMin() const { return m_yMinSpinBox->value(); }
void ecvAxesGridDialog::setYMin(double value) {
    m_yMinSpinBox->setValue(value);
}
double ecvAxesGridDialog::getYMax() const { return m_yMaxSpinBox->value(); }
void ecvAxesGridDialog::setYMax(double value) {
    m_yMaxSpinBox->setValue(value);
}
double ecvAxesGridDialog::getZMin() const { return m_zMinSpinBox->value(); }
void ecvAxesGridDialog::setZMin(double value) {
    m_zMinSpinBox->setValue(value);
}
double ecvAxesGridDialog::getZMax() const { return m_zMaxSpinBox->value(); }
void ecvAxesGridDialog::setZMax(double value) {
    m_zMaxSpinBox->setValue(value);
}
