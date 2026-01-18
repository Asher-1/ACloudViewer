// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "eCV_db.h"

// Qt
#include <QColor>
#include <QWidget>

class QToolButton;

namespace Ui {
class ecvFontPropertyWidget;
}

/**
 * @brief A reusable font property widget matching ParaView's font editor style.
 *
 * This widget provides controls for:
 * - Font Family (Arial, Courier, Times)
 * - Font Size (1-999)
 * - Font Color (optional)
 * - Opacity (0.0-1.0)
 * - Bold toggle
 * - Italic toggle
 * - Shadow toggle
 *
 * The widget can be used with or without color picker. When color picker
 * is hidden, only font styling properties are shown.
 *
 * Usage:
 * @code
 * ecvFontPropertyWidget* fontWidget = new ecvFontPropertyWidget(this);
 * connect(fontWidget, &ecvFontPropertyWidget::fontPropertiesChanged,
 *         this, &MyClass::onFontChanged);
 *
 * // Optionally hide color picker
 * fontWidget->setColorPickerVisible(false);
 * @endcode
 */
class ECV_DB_LIB_API ecvFontPropertyWidget : public QWidget {
    Q_OBJECT

public:
    //! Font property structure for convenience
    struct FontProperties {
        QString family = "Arial";
        int size = 18;
        QColor color = QColor(255, 255, 255);  // Default white
        double opacity = 1.0;
        bool bold = false;
        bool italic = false;
        bool shadow = true;
        QString horizontalJustification = "Left";  // Left, Center, Right
        QString verticalJustification = "Bottom";  // Top, Center, Bottom

        //! Returns VTK font family index (0=Arial, 1=Courier, 2=Times)
        int fontFamilyIndex() const;

        //! Sets family from VTK font family index
        void setFontFamilyFromIndex(int index);
    };

    explicit ecvFontPropertyWidget(QWidget* parent = nullptr);
    ~ecvFontPropertyWidget() override;

    ///@{
    //! Get/Set all font properties at once
    FontProperties fontProperties() const;
    void setFontProperties(const FontProperties& props);
    ///@}

    ///@{
    //! Individual property getters
    QString fontFamily() const;
    int fontFamilyIndex() const;
    int fontSize() const;
    QColor fontColor() const;
    double fontOpacity() const;
    bool isBold() const;
    bool isItalic() const;
    bool hasShadow() const;
    QString horizontalJustification() const;
    QString verticalJustification() const;
    ///@}

    ///@{
    //! Individual property setters (with signal blocking option)
    void setFontFamily(const QString& family, bool blockSignal = false);
    void setFontFamilyIndex(int index, bool blockSignal = false);
    void setFontSize(int size, bool blockSignal = false);
    void setFontColor(const QColor& color, bool blockSignal = false);
    void setFontOpacity(double opacity, bool blockSignal = false);
    void setBold(bool bold, bool blockSignal = false);
    void setItalic(bool italic, bool blockSignal = false);
    void setShadow(bool shadow, bool blockSignal = false);
    void setHorizontalJustification(const QString& justification,
                                    bool blockSignal = false);
    void setVerticalJustification(const QString& justification,
                                  bool blockSignal = false);
    ///@}

    //! Show/hide the color picker button
    void setColorPickerVisible(bool visible);

    //! Enable/disable all controls
    void setControlsEnabled(bool enabled);

Q_SIGNALS:
    //! Emitted when any font property changes
    void fontPropertiesChanged();

    //! Individual property change signals
    void fontFamilyChanged(const QString& family);
    void fontFamilyIndexChanged(int index);
    void fontSizeChanged(int size);
    void fontColorChanged(const QColor& color);
    void fontOpacityChanged(double opacity);
    void boldChanged(bool bold);
    void italicChanged(bool italic);
    void shadowChanged(bool shadow);
    void horizontalJustificationChanged(const QString& justification);
    void verticalJustificationChanged(const QString& justification);

private Q_SLOTS:
    void onFontFamilyChanged(int index);
    void onFontSizeChanged(int size);
    void onFontColorClicked();
    void onFontOpacityChanged(double opacity);
    void onBoldToggled(bool checked);
    void onItalicToggled(bool checked);
    void onShadowToggled(bool checked);
    void onHorizontalJustificationTriggered(QAction* action);
    void onVerticalJustificationTriggered(QAction* action);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void setupConnections();
    void updateColorButtonAppearance();
    void setupHorizontalJustificationButton();
    void setupVerticalJustificationButton();
    void updateJustificationButtonIcon(const QString& justification,
                                       QToolButton* button);

    Ui::ecvFontPropertyWidget* ui;

    //! Current font color
    QColor m_fontColor;

    //! Current justification values
    QString m_horizontalJustification = "Left";
    QString m_verticalJustification = "Bottom";

    //! Flag to prevent signal emission during programmatic updates
    bool m_blockSignals = false;
};
