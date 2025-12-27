// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvSelectionLabelPropertiesDialog.h"

#include <widgets/ecvFontPropertyWidget.h>

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

//-----------------------------------------------------------------------------
cvSelectionLabelPropertiesDialog::cvSelectionLabelPropertiesDialog(
        QWidget* parent, bool isInteractive)
    : QDialog(parent), m_isInteractive(isInteractive) {
    setWindowTitle(isInteractive ? tr("Interactive Selection Label Properties")
                                 : tr("Selection Label Properties"));
    setMinimumWidth(480);
    loadDefaults();
    setupUi();
}

//-----------------------------------------------------------------------------
cvSelectionLabelPropertiesDialog::~cvSelectionLabelPropertiesDialog() {}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::loadDefaults() {
    m_defaultProperties = LabelProperties();
    m_properties = m_defaultProperties;
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::setupUi() {
    QVBoxLayout* mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(10);

    // === General Settings ===
    QFormLayout* generalLayout = new QFormLayout();
    generalLayout->setSpacing(8);

    // Opacity
    QHBoxLayout* opacityLayout = new QHBoxLayout();
    m_opacitySlider = new QSlider(Qt::Horizontal);
    m_opacitySlider->setRange(0, 100);
    m_opacitySlider->setValue(static_cast<int>(m_properties.opacity * 100));
    m_opacitySpin = new QDoubleSpinBox();
    m_opacitySpin->setRange(0.0, 1.0);
    m_opacitySpin->setSingleStep(0.1);
    m_opacitySpin->setDecimals(2);
    m_opacitySpin->setValue(m_properties.opacity);
    opacityLayout->addWidget(m_opacitySlider, 1);
    opacityLayout->addWidget(m_opacitySpin);
    connect(m_opacitySlider, &QSlider::valueChanged, this,
            &cvSelectionLabelPropertiesDialog::onOpacitySliderChanged);
    connect(m_opacitySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            [this](double value) {
                m_opacitySlider->blockSignals(true);
                m_opacitySlider->setValue(static_cast<int>(value * 100));
                m_opacitySlider->blockSignals(false);
                m_properties.opacity = value;
            });
    generalLayout->addRow(tr("Opacity"), opacityLayout);

    // Point Size
    m_pointSizeSpin = new QSpinBox();
    m_pointSizeSpin->setRange(1, 50);
    m_pointSizeSpin->setValue(m_properties.pointSize);
    connect(m_pointSizeSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int value) { m_properties.pointSize = value; });
    generalLayout->addRow(tr("Point Size"), m_pointSizeSpin);

    // Line Width
    m_lineWidthSpin = new QSpinBox();
    m_lineWidthSpin->setRange(1, 20);
    m_lineWidthSpin->setValue(m_properties.lineWidth);
    connect(m_lineWidthSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int value) { m_properties.lineWidth = value; });
    generalLayout->addRow(tr("Line Width"), m_lineWidthSpin);

    mainLayout->addLayout(generalLayout);

    // === Cell Label Font ===
    QGroupBox* cellFontGroup = new QGroupBox(tr("Cell Label Font"));
    QVBoxLayout* cellFontLayout = new QVBoxLayout();

    // Font property widget for cell labels
    m_cellFontWidget = new ecvFontPropertyWidget(this);
    m_cellFontWidget->setFontProperties(
            labelPropertiesToFontProperties(m_properties, true));
    connect(m_cellFontWidget, &ecvFontPropertyWidget::fontPropertiesChanged,
            this,
            &cvSelectionLabelPropertiesDialog::onCellFontPropertiesChanged);
    cellFontLayout->addWidget(m_cellFontWidget);

    // Cell Label Format
    QHBoxLayout* cellFormatLayout = new QHBoxLayout();
    QLabel* cellFormatLabel = new QLabel(tr("Cell Label Format"));
    m_cellFormatEdit = new QLineEdit();
    m_cellFormatEdit->setText(m_properties.cellLabelFormat);
    m_cellFormatEdit->setPlaceholderText(tr("e.g., ID: %s"));
    connect(m_cellFormatEdit, &QLineEdit::textChanged,
            [this](const QString& text) {
                m_properties.cellLabelFormat = text;
            });
    cellFormatLayout->addWidget(cellFormatLabel);
    cellFormatLayout->addWidget(m_cellFormatEdit, 1);
    cellFontLayout->addLayout(cellFormatLayout);

    cellFontGroup->setLayout(cellFontLayout);
    mainLayout->addWidget(cellFontGroup);

    // === Point Label Font ===
    QGroupBox* pointFontGroup = new QGroupBox(tr("Point Label Font"));
    QVBoxLayout* pointFontLayout = new QVBoxLayout();

    // Font property widget for point labels
    m_pointFontWidget = new ecvFontPropertyWidget(this);
    m_pointFontWidget->setFontProperties(
            labelPropertiesToFontProperties(m_properties, false));
    connect(m_pointFontWidget, &ecvFontPropertyWidget::fontPropertiesChanged,
            this,
            &cvSelectionLabelPropertiesDialog::onPointFontPropertiesChanged);
    pointFontLayout->addWidget(m_pointFontWidget);

    // Point Label Format
    QHBoxLayout* pointFormatLayout = new QHBoxLayout();
    QLabel* pointFormatLabel = new QLabel(tr("Point Label Format"));
    m_pointFormatEdit = new QLineEdit();
    m_pointFormatEdit->setText(m_properties.pointLabelFormat);
    m_pointFormatEdit->setPlaceholderText(tr("e.g., ID: %s"));
    connect(m_pointFormatEdit, &QLineEdit::textChanged,
            [this](const QString& text) {
                m_properties.pointLabelFormat = text;
            });
    pointFormatLayout->addWidget(pointFormatLabel);
    pointFormatLayout->addWidget(m_pointFormatEdit, 1);
    pointFontLayout->addLayout(pointFormatLayout);

    pointFontGroup->setLayout(pointFontLayout);
    mainLayout->addWidget(pointFontGroup);

    // === Tooltip Settings ===
    QGroupBox* tooltipGroup = new QGroupBox(tr("Tooltip Settings"));
    QFormLayout* tooltipLayout = new QFormLayout();
    tooltipLayout->setSpacing(8);

    m_showTooltipsCheckBox = new QCheckBox(tr("Show tooltips on hover"));
    m_showTooltipsCheckBox->setChecked(m_properties.showTooltips);
    connect(m_showTooltipsCheckBox, &QCheckBox::toggled,
            [this](bool checked) { m_properties.showTooltips = checked; });
    tooltipLayout->addRow(m_showTooltipsCheckBox);

    m_maxTooltipAttributesSpin = new QSpinBox();
    m_maxTooltipAttributesSpin->setRange(1, 50);
    m_maxTooltipAttributesSpin->setValue(m_properties.maxTooltipAttributes);
    connect(m_maxTooltipAttributesSpin,
            QOverload<int>::of(&QSpinBox::valueChanged),
            [this](int value) { m_properties.maxTooltipAttributes = value; });
    tooltipLayout->addRow(tr("Max attributes:"), m_maxTooltipAttributesSpin);

    tooltipGroup->setLayout(tooltipLayout);
    mainLayout->addWidget(tooltipGroup);

    // === Dialog Buttons ===
    QHBoxLayout* buttonLayout = new QHBoxLayout();

    // Left side buttons
    m_refreshButton = new QToolButton();
    m_refreshButton->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
    m_refreshButton->setToolTip(tr("Refresh from settings"));
    buttonLayout->addWidget(m_refreshButton);

    m_saveButton = new QToolButton();
    m_saveButton->setIcon(style()->standardIcon(QStyle::SP_DialogSaveButton));
    m_saveButton->setToolTip(tr("Save as default"));
    buttonLayout->addWidget(m_saveButton);

    buttonLayout->addStretch();

    // Right side buttons
    m_applyButton = new QPushButton(tr("Apply"));
    m_applyButton->setIcon(style()->standardIcon(QStyle::SP_DialogApplyButton));
    connect(m_applyButton, &QPushButton::clicked, this,
            &cvSelectionLabelPropertiesDialog::onApplyClicked);
    buttonLayout->addWidget(m_applyButton);

    m_resetButton = new QPushButton(tr("Reset"));
    m_resetButton->setIcon(style()->standardIcon(QStyle::SP_DialogResetButton));
    connect(m_resetButton, &QPushButton::clicked, this,
            &cvSelectionLabelPropertiesDialog::onResetClicked);
    buttonLayout->addWidget(m_resetButton);

    m_cancelButton = new QPushButton(tr("Cancel"));
    m_cancelButton->setIcon(
            style()->standardIcon(QStyle::SP_DialogCancelButton));
    connect(m_cancelButton, &QPushButton::clicked, this, &QDialog::reject);
    buttonLayout->addWidget(m_cancelButton);

    m_okButton = new QPushButton(tr("OK"));
    m_okButton->setIcon(style()->standardIcon(QStyle::SP_DialogOkButton));
    m_okButton->setDefault(true);
    connect(m_okButton, &QPushButton::clicked, [this]() {
        onApplyClicked();
        accept();
    });
    buttonLayout->addWidget(m_okButton);

    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::setProperties(
        const LabelProperties& props) {
    m_properties = props;

    // Update UI
    m_opacitySlider->setValue(static_cast<int>(props.opacity * 100));
    m_opacitySpin->setValue(props.opacity);
    m_pointSizeSpin->setValue(props.pointSize);
    m_lineWidthSpin->setValue(props.lineWidth);

    // Cell font
    if (m_cellFontWidget) {
        m_cellFontWidget->setFontProperties(
                labelPropertiesToFontProperties(props, true));
    }
    m_cellFormatEdit->setText(props.cellLabelFormat);

    // Point font
    if (m_pointFontWidget) {
        m_pointFontWidget->setFontProperties(
                labelPropertiesToFontProperties(props, false));
    }
    m_pointFormatEdit->setText(props.pointLabelFormat);

    // Tooltip settings
    m_showTooltipsCheckBox->setChecked(props.showTooltips);
    m_maxTooltipAttributesSpin->setValue(props.maxTooltipAttributes);
}

//-----------------------------------------------------------------------------
cvSelectionLabelPropertiesDialog::LabelProperties
cvSelectionLabelPropertiesDialog::properties() const {
    return m_properties;
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::updatePropertiesFromWidgets() {
    // Cell font properties
    if (m_cellFontWidget) {
        fontPropertiesToLabelProperties(m_cellFontWidget->fontProperties(),
                                        m_properties, true);
    }

    // Point font properties
    if (m_pointFontWidget) {
        fontPropertiesToLabelProperties(m_pointFontWidget->fontProperties(),
                                        m_properties, false);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::onApplyClicked() {
    updatePropertiesFromWidgets();
    Q_EMIT propertiesApplied(m_properties);
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::onResetClicked() {
    setProperties(m_defaultProperties);
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::onOpacitySliderChanged(int value) {
    double opacity = value / 100.0;
    m_opacitySpin->blockSignals(true);
    m_opacitySpin->setValue(opacity);
    m_opacitySpin->blockSignals(false);
    m_properties.opacity = opacity;
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::onCellFontPropertiesChanged() {
    if (m_cellFontWidget) {
        fontPropertiesToLabelProperties(m_cellFontWidget->fontProperties(),
                                        m_properties, true);
    }
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::onPointFontPropertiesChanged() {
    if (m_pointFontWidget) {
        fontPropertiesToLabelProperties(m_pointFontWidget->fontProperties(),
                                        m_properties, false);
    }
}

//-----------------------------------------------------------------------------
ecvFontPropertyWidget::FontProperties
cvSelectionLabelPropertiesDialog::labelPropertiesToFontProperties(
        const LabelProperties& props, bool isCellLabel) {
    ecvFontPropertyWidget::FontProperties fontProps;
    if (isCellLabel) {
        fontProps.family = props.cellLabelFontFamily;
        fontProps.size = props.cellLabelFontSize;
        fontProps.color = props.cellLabelColor;
        fontProps.opacity = props.cellLabelOpacity;
        fontProps.bold = props.cellLabelBold;
        fontProps.italic = props.cellLabelItalic;
        fontProps.shadow = props.cellLabelShadow;
        fontProps.horizontalJustification =
                props.cellLabelHorizontalJustification;
        fontProps.verticalJustification = props.cellLabelVerticalJustification;
    } else {
        fontProps.family = props.pointLabelFontFamily;
        fontProps.size = props.pointLabelFontSize;
        fontProps.color = props.pointLabelColor;
        fontProps.opacity = props.pointLabelOpacity;
        fontProps.bold = props.pointLabelBold;
        fontProps.italic = props.pointLabelItalic;
        fontProps.shadow = props.pointLabelShadow;
        fontProps.horizontalJustification =
                props.pointLabelHorizontalJustification;
        fontProps.verticalJustification = props.pointLabelVerticalJustification;
    }
    return fontProps;
}

//-----------------------------------------------------------------------------
void cvSelectionLabelPropertiesDialog::fontPropertiesToLabelProperties(
        const ecvFontPropertyWidget::FontProperties& fontProps,
        LabelProperties& props,
        bool isCellLabel) {
    if (isCellLabel) {
        props.cellLabelFontFamily = fontProps.family;
        props.cellLabelFontSize = fontProps.size;
        props.cellLabelColor = fontProps.color;
        props.cellLabelOpacity = fontProps.opacity;
        props.cellLabelBold = fontProps.bold;
        props.cellLabelItalic = fontProps.italic;
        props.cellLabelShadow = fontProps.shadow;
        props.cellLabelHorizontalJustification =
                fontProps.horizontalJustification;
        props.cellLabelVerticalJustification = fontProps.verticalJustification;
    } else {
        props.pointLabelFontFamily = fontProps.family;
        props.pointLabelFontSize = fontProps.size;
        props.pointLabelColor = fontProps.color;
        props.pointLabelOpacity = fontProps.opacity;
        props.pointLabelBold = fontProps.bold;
        props.pointLabelItalic = fontProps.italic;
        props.pointLabelShadow = fontProps.shadow;
        props.pointLabelHorizontalJustification =
                fontProps.horizontalJustification;
        props.pointLabelVerticalJustification = fontProps.verticalJustification;
    }
}
