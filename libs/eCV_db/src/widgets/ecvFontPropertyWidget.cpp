// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "widgets/ecvFontPropertyWidget.h"
#include "ui_ecvFontPropertyWidget.h"

#include <QActionGroup>
#include <QColorDialog>
#include <QIcon>
#include <QMenu>
#include <QToolButton>

//-----------------------------------------------------------------------------
// FontProperties helper methods
//-----------------------------------------------------------------------------
int ecvFontPropertyWidget::FontProperties::fontFamilyIndex() const {
    if (family == "Arial") return 0;
    if (family == "Courier") return 1;
    if (family == "Times") return 2;
    return 0;  // Default to Arial
}

void ecvFontPropertyWidget::FontProperties::setFontFamilyFromIndex(int index) {
    switch (index) {
        case 0:
            family = "Arial";
            break;
        case 1:
            family = "Courier";
            break;
        case 2:
            family = "Times";
            break;
        default:
            family = "Arial";
            break;
    }
}

//-----------------------------------------------------------------------------
// ecvFontPropertyWidget implementation
//-----------------------------------------------------------------------------
ecvFontPropertyWidget::ecvFontPropertyWidget(QWidget* parent)
    : QWidget(parent),
      ui(new Ui::ecvFontPropertyWidget),
      m_fontColor(255, 255, 255) {
    ui->setupUi(this);
    
    // Ensure icons are loaded (fallback if UI file doesn't load them)
    // Icons are from ParaView: pqBold24.png, pqItalics24.png, pqShadow24.png
    if (ui->boldButton) {
        QIcon boldIcon(":/Resources/images/font/pqBold24.png");
        if (!boldIcon.isNull()) {
            ui->boldButton->setIcon(boldIcon);
        }
    }
    if (ui->italicButton) {
        QIcon italicIcon(":/Resources/images/font/pqItalics24.png");
        if (!italicIcon.isNull()) {
            ui->italicButton->setIcon(italicIcon);
        }
    }
    if (ui->shadowButton) {
        QIcon shadowIcon(":/Resources/images/font/pqShadow24.png");
        if (!shadowIcon.isNull()) {
            ui->shadowButton->setIcon(shadowIcon);
        }
    }
    
    setupConnections();
    updateColorButtonAppearance();
    
    // Apply ParaView-style toggle button styling for better visual feedback
    const QString toggleButtonStyle =
        "QToolButton {"
        "    border: 1px solid #999;"
        "    border-radius: 3px;"
        "    padding: 2px;"
        "    background-color: #f0f0f0;"
        "}"
        "QToolButton:hover {"
        "    background-color: #e0e0e0;"
        "    border-color: #666;"
        "}"
        "QToolButton:checked {"
        "    background-color: #c0d0e8;"
        "    border-color: #4080c0;"
        "}"
        "QToolButton:checked:hover {"
        "    background-color: #a0c0e0;"
        "}";
    
    if (ui->boldButton) ui->boldButton->setStyleSheet(toggleButtonStyle);
    if (ui->italicButton) ui->italicButton->setStyleSheet(toggleButtonStyle);
    if (ui->shadowButton) ui->shadowButton->setStyleSheet(toggleButtonStyle);
    
    // Setup justification buttons
    setupHorizontalJustificationButton();
    setupVerticalJustificationButton();
}

ecvFontPropertyWidget::~ecvFontPropertyWidget() {
    delete ui;
}

void ecvFontPropertyWidget::setupConnections() {
    connect(ui->fontFamilyComboBox,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &ecvFontPropertyWidget::onFontFamilyChanged);
    connect(ui->fontSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ecvFontPropertyWidget::onFontSizeChanged);
    connect(ui->fontColorButton, &QPushButton::clicked, this,
            &ecvFontPropertyWidget::onFontColorClicked);
    connect(ui->fontOpacitySpinBox,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            &ecvFontPropertyWidget::onFontOpacityChanged);
    connect(ui->boldButton, &QToolButton::toggled, this,
            &ecvFontPropertyWidget::onBoldToggled);
    connect(ui->italicButton, &QToolButton::toggled, this,
            &ecvFontPropertyWidget::onItalicToggled);
    connect(ui->shadowButton, &QToolButton::toggled, this,
            &ecvFontPropertyWidget::onShadowToggled);
}

ecvFontPropertyWidget::FontProperties
ecvFontPropertyWidget::fontProperties() const {
    FontProperties props;
    props.family = fontFamily();
    props.size = fontSize();
    props.color = fontColor();
    props.opacity = fontOpacity();
    props.bold = isBold();
    props.italic = isItalic();
    props.shadow = hasShadow();
    props.horizontalJustification = horizontalJustification();
    props.verticalJustification = verticalJustification();
    return props;
}

void ecvFontPropertyWidget::setFontProperties(const FontProperties& props) {
    m_blockSignals = true;
    setFontFamily(props.family, true);
    setFontSize(props.size, true);
    setFontColor(props.color, true);
    setFontOpacity(props.opacity, true);
    setBold(props.bold, true);
    setItalic(props.italic, true);
    setShadow(props.shadow, true);
    setHorizontalJustification(props.horizontalJustification, true);
    setVerticalJustification(props.verticalJustification, true);
    m_blockSignals = false;

    // Emit single change signal
    Q_EMIT fontPropertiesChanged();
}

QString ecvFontPropertyWidget::fontFamily() const {
    return ui->fontFamilyComboBox ? ui->fontFamilyComboBox->currentText()
                                  : "Arial";
}

int ecvFontPropertyWidget::fontFamilyIndex() const {
    return ui->fontFamilyComboBox ? ui->fontFamilyComboBox->currentIndex() : 0;
}

int ecvFontPropertyWidget::fontSize() const {
    return ui->fontSizeSpinBox ? ui->fontSizeSpinBox->value() : 6;
}

QColor ecvFontPropertyWidget::fontColor() const { return m_fontColor; }

double ecvFontPropertyWidget::fontOpacity() const {
    return ui->fontOpacitySpinBox ? ui->fontOpacitySpinBox->value() : 1.0;
}

bool ecvFontPropertyWidget::isBold() const {
    return ui->boldButton ? ui->boldButton->isChecked() : false;
}

bool ecvFontPropertyWidget::isItalic() const {
    return ui->italicButton ? ui->italicButton->isChecked() : false;
}

bool ecvFontPropertyWidget::hasShadow() const {
    return ui->shadowButton ? ui->shadowButton->isChecked() : true;
}

QString ecvFontPropertyWidget::horizontalJustification() const {
    return m_horizontalJustification;
}

QString ecvFontPropertyWidget::verticalJustification() const {
    return m_verticalJustification;
}

void ecvFontPropertyWidget::setFontFamily(const QString& family,
                                          bool blockSignal) {
    if (!ui->fontFamilyComboBox) return;

    int index = ui->fontFamilyComboBox->findText(family);
    if (index >= 0) {
        if (blockSignal) {
            ui->fontFamilyComboBox->blockSignals(true);
        }
        ui->fontFamilyComboBox->setCurrentIndex(index);
        if (blockSignal) {
            ui->fontFamilyComboBox->blockSignals(false);
        }
    }
}

void ecvFontPropertyWidget::setFontFamilyIndex(int index, bool blockSignal) {
    if (!ui->fontFamilyComboBox) return;

    if (index >= 0 && index < ui->fontFamilyComboBox->count()) {
        if (blockSignal) {
            ui->fontFamilyComboBox->blockSignals(true);
        }
        ui->fontFamilyComboBox->setCurrentIndex(index);
        if (blockSignal) {
            ui->fontFamilyComboBox->blockSignals(false);
        }
    }
}

void ecvFontPropertyWidget::setFontSize(int size, bool blockSignal) {
    if (!ui->fontSizeSpinBox) return;

    if (blockSignal) {
        ui->fontSizeSpinBox->blockSignals(true);
    }
    ui->fontSizeSpinBox->setValue(size);
    if (blockSignal) {
        ui->fontSizeSpinBox->blockSignals(false);
    }
}

void ecvFontPropertyWidget::setFontColor(const QColor& color, bool blockSignal) {
    if (m_fontColor != color) {
        m_fontColor = color;
        updateColorButtonAppearance();
        if (!blockSignal && !m_blockSignals) {
            Q_EMIT fontColorChanged(color);
            Q_EMIT fontPropertiesChanged();
        }
    }
}

void ecvFontPropertyWidget::setFontOpacity(double opacity, bool blockSignal) {
    if (!ui->fontOpacitySpinBox) return;

    if (blockSignal) {
        ui->fontOpacitySpinBox->blockSignals(true);
    }
    ui->fontOpacitySpinBox->setValue(opacity);
    if (blockSignal) {
        ui->fontOpacitySpinBox->blockSignals(false);
    }
}

void ecvFontPropertyWidget::setBold(bool bold, bool blockSignal) {
    if (!ui->boldButton) return;

    if (blockSignal) {
        ui->boldButton->blockSignals(true);
    }
    ui->boldButton->setChecked(bold);
    if (blockSignal) {
        ui->boldButton->blockSignals(false);
    }
}

void ecvFontPropertyWidget::setItalic(bool italic, bool blockSignal) {
    if (!ui->italicButton) return;

    if (blockSignal) {
        ui->italicButton->blockSignals(true);
    }
    ui->italicButton->setChecked(italic);
    if (blockSignal) {
        ui->italicButton->blockSignals(false);
    }
}

void ecvFontPropertyWidget::setShadow(bool shadow, bool blockSignal) {
    if (!ui->shadowButton) return;

    if (blockSignal) {
        ui->shadowButton->blockSignals(true);
    }
    ui->shadowButton->setChecked(shadow);
    if (blockSignal) {
        ui->shadowButton->blockSignals(false);
    }
}

void ecvFontPropertyWidget::setHorizontalJustification(const QString& justification, bool blockSignal) {
    if (m_horizontalJustification != justification) {
        m_horizontalJustification = justification;
        updateJustificationButtonIcon(justification, ui->horizontalJustificationButton);
        if (!blockSignal && !m_blockSignals) {
            Q_EMIT horizontalJustificationChanged(justification);
            Q_EMIT fontPropertiesChanged();
        }
    }
}

void ecvFontPropertyWidget::setVerticalJustification(const QString& justification, bool blockSignal) {
    if (m_verticalJustification != justification) {
        m_verticalJustification = justification;
        updateJustificationButtonIcon(justification, ui->verticalJustificationButton);
        if (!blockSignal && !m_blockSignals) {
            Q_EMIT verticalJustificationChanged(justification);
            Q_EMIT fontPropertiesChanged();
        }
    }
}

void ecvFontPropertyWidget::setColorPickerVisible(bool visible) {
    if (ui->fontColorButton) {
        ui->fontColorButton->setVisible(visible);
    }
}

void ecvFontPropertyWidget::setControlsEnabled(bool enabled) {
    if (ui->fontFamilyComboBox) ui->fontFamilyComboBox->setEnabled(enabled);
    if (ui->fontSizeSpinBox) ui->fontSizeSpinBox->setEnabled(enabled);
    if (ui->fontColorButton) ui->fontColorButton->setEnabled(enabled);
    if (ui->fontOpacitySpinBox) ui->fontOpacitySpinBox->setEnabled(enabled);
    if (ui->boldButton) ui->boldButton->setEnabled(enabled);
    if (ui->italicButton) ui->italicButton->setEnabled(enabled);
    if (ui->shadowButton) ui->shadowButton->setEnabled(enabled);
}

void ecvFontPropertyWidget::onFontFamilyChanged(int index) {
    if (!m_blockSignals) {
        Q_EMIT fontFamilyChanged(fontFamily());
        Q_EMIT fontFamilyIndexChanged(index);
        Q_EMIT fontPropertiesChanged();
    }
}

void ecvFontPropertyWidget::onFontSizeChanged(int size) {
    if (!m_blockSignals) {
        Q_EMIT fontSizeChanged(size);
        Q_EMIT fontPropertiesChanged();
    }
}

void ecvFontPropertyWidget::onFontColorClicked() {
    QColor newColor =
            QColorDialog::getColor(m_fontColor, this, tr("Select Font Color"));
    if (newColor.isValid() && newColor != m_fontColor) {
        setFontColor(newColor, false);
    }
}

void ecvFontPropertyWidget::onFontOpacityChanged(double opacity) {
    if (!m_blockSignals) {
        Q_EMIT fontOpacityChanged(opacity);
        Q_EMIT fontPropertiesChanged();
    }
}

void ecvFontPropertyWidget::onBoldToggled(bool checked) {
    if (!m_blockSignals) {
        Q_EMIT boldChanged(checked);
        Q_EMIT fontPropertiesChanged();
    }
}

void ecvFontPropertyWidget::onItalicToggled(bool checked) {
    if (!m_blockSignals) {
        Q_EMIT italicChanged(checked);
        Q_EMIT fontPropertiesChanged();
    }
}

void ecvFontPropertyWidget::onShadowToggled(bool checked) {
    if (!m_blockSignals) {
        Q_EMIT shadowChanged(checked);
        Q_EMIT fontPropertiesChanged();
    }
}

void ecvFontPropertyWidget::updateColorButtonAppearance() {
    if (ui->fontColorButton) {
        QString styleSheet =
                QString("QPushButton { background-color: rgb(%1, %2, %3); "
                        "border: 1px solid gray; }")
                        .arg(m_fontColor.red())
                        .arg(m_fontColor.green())
                        .arg(m_fontColor.blue());
        ui->fontColorButton->setStyleSheet(styleSheet);
    }
}

void ecvFontPropertyWidget::setupHorizontalJustificationButton() {
    if (!ui->horizontalJustificationButton) return;
    
    QActionGroup* actionGroup = new QActionGroup(this);
    actionGroup->setExclusive(true);
    
    // Create actions with icons, matching ParaView's implementation
    QAction* leftAlign = new QAction(QIcon(":/Resources/images/font/pqTextAlignLeft.svg"), tr("Left"), actionGroup);
    leftAlign->setIconVisibleInMenu(true);
    
    QAction* centerAlign = new QAction(QIcon(":/Resources/images/font/pqTextAlignCenter.svg"), tr("Center"), actionGroup);
    centerAlign->setIconVisibleInMenu(true);
    
    QAction* rightAlign = new QAction(QIcon(":/Resources/images/font/pqTextAlignRight.svg"), tr("Right"), actionGroup);
    rightAlign->setIconVisibleInMenu(true);
    
    QMenu* popup = new QMenu(this);
    popup->addAction(leftAlign);
    popup->addAction(centerAlign);
    popup->addAction(rightAlign);
    ui->horizontalJustificationButton->setMenu(popup);
    
    connect(actionGroup, &QActionGroup::triggered, this,
            &ecvFontPropertyWidget::onHorizontalJustificationTriggered);
    
    // Set initial icon
    updateJustificationButtonIcon("Left", ui->horizontalJustificationButton);
}

void ecvFontPropertyWidget::setupVerticalJustificationButton() {
    if (!ui->verticalJustificationButton) return;
    
    QActionGroup* actionGroup = new QActionGroup(this);
    actionGroup->setExclusive(true);
    
    // Create actions with icons, matching ParaView's implementation
    QAction* topAlign = new QAction(QIcon(":/Resources/images/font/pqTextVerticalAlignTop.svg"), tr("Top"), actionGroup);
    topAlign->setIconVisibleInMenu(true);
    
    QAction* centerAlign = new QAction(QIcon(":/Resources/images/font/pqTextVerticalAlignCenter.svg"), tr("Center"), actionGroup);
    centerAlign->setIconVisibleInMenu(true);
    
    QAction* bottomAlign = new QAction(QIcon(":/Resources/images/font/pqTextVerticalAlignBottom.svg"), tr("Bottom"), actionGroup);
    bottomAlign->setIconVisibleInMenu(true);
    
    QMenu* popup = new QMenu(this);
    popup->addAction(topAlign);
    popup->addAction(centerAlign);
    popup->addAction(bottomAlign);
    ui->verticalJustificationButton->setMenu(popup);
    
    connect(actionGroup, &QActionGroup::triggered, this,
            &ecvFontPropertyWidget::onVerticalJustificationTriggered);
    
    // Set initial icon
    updateJustificationButtonIcon("Bottom", ui->verticalJustificationButton);
}

void ecvFontPropertyWidget::updateJustificationButtonIcon(const QString& justification, QToolButton* button) {
    if (!button || !button->menu()) return;
    
    QList<QAction*> actions = button->menu()->actions();
    for (QAction* action : actions) {
        if (action->text() == justification) {
            QIcon icon = action->icon();
            if (!icon.isNull()) {
                button->setIcon(icon);
            } else {
                // Fallback: use text as icon
                button->setText(justification.left(1));
            }
            break;
        }
    }
}

void ecvFontPropertyWidget::onHorizontalJustificationTriggered(QAction* action) {
    QString justification = action->text();
    setHorizontalJustification(justification, false);
}

void ecvFontPropertyWidget::onVerticalJustificationTriggered(QAction* action) {
    QString justification = action->text();
    setVerticalJustification(justification, false);
}
