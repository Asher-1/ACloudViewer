// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_AXES_GRID_DIALOG_H
#define ECV_AXES_GRID_DIALOG_H

#include <QColor>
#include <QDialog>
#include <QList>
#include <QPair>

class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;
class QPushButton;
class QLabel;
class QLineEdit;
class QGroupBox;
class ecvCustomLabelsWidget;

/**
 * @brief ParaView-style Axes Grid Properties Dialog
 *
 * Complete implementation matching ParaView's GridAxes3DActor properties panel.
 * Features:
 * - Title Texts (X/Y/Z)
 * - Face Properties (Grid Color, Show Grid)
 * - X/Y/Z Axis Label Properties with Custom Labels support
 * - Bounds (Use Custom Bounds)
 * - Real-time preview via Apply button
 * - Non-modal dialog
 */
class ecvAxesGridDialog : public QDialog {
    Q_OBJECT

public:
    explicit ecvAxesGridDialog(const QString& title, QWidget* parent = nullptr);
    ~ecvAxesGridDialog() override;

    // ========================================================================
    // Title Texts (Section 1)
    // ========================================================================
    QString getXTitle() const;
    void setXTitle(const QString& title);
    QString getYTitle() const;
    void setYTitle(const QString& title);
    QString getZTitle() const;
    void setZTitle(const QString& title);

    // ========================================================================
    // Face Properties (Section 2)
    // ========================================================================
    QColor getGridColor() const;
    void setGridColor(const QColor& color);
    bool getShowGrid() const;
    void setShowGrid(bool show);

    // ========================================================================
    // X Axis Label Properties (Section 3)
    // ========================================================================
    bool getXAxisUseCustomLabels() const;
    void setXAxisUseCustomLabels(bool use);
    QList<QPair<double, QString>> getXAxisCustomLabels() const;
    void setXAxisCustomLabels(const QList<QPair<double, QString>>& labels);

    // ========================================================================
    // Y Axis Label Properties (Section 4)
    // ========================================================================
    bool getYAxisUseCustomLabels() const;
    void setYAxisUseCustomLabels(bool use);
    QList<QPair<double, QString>> getYAxisCustomLabels() const;
    void setYAxisCustomLabels(const QList<QPair<double, QString>>& labels);

    // ========================================================================
    // Z Axis Label Properties (Section 5)
    // ========================================================================
    bool getZAxisUseCustomLabels() const;
    void setZAxisUseCustomLabels(bool use);
    QList<QPair<double, QString>> getZAxisCustomLabels() const;
    void setZAxisCustomLabels(const QList<QPair<double, QString>>& labels);

    // ========================================================================
    // Bounds (Section 6)
    // ========================================================================
    bool getUseCustomBounds() const;
    void setUseCustomBounds(bool use);

    double getXMin() const;
    void setXMin(double value);
    double getXMax() const;
    void setXMax(double value);
    double getYMin() const;
    void setYMin(double value);
    double getYMax() const;
    void setYMax(double value);
    double getZMin() const;
    void setZMin(double value);
    double getZMax() const;
    void setZMax(double value);

    // ========================================================================
    // Legacy compatibility (for backward compatibility with old interface)
    // ========================================================================
    QColor getColor() const { return getGridColor(); }
    void setColor(const QColor& color) { setGridColor(color); }
    double getLineWidth() const { return 1.0; }
    void setLineWidth(double) {}
    double getOpacity() const { return 1.0; }
    void setOpacity(double) {}
    bool getShowLabels() const { return true; }
    void setShowLabels(bool) {}

signals:
    void propertiesChanged();
    void applyRequested();  // Real-time preview signal

private slots:
    void onColorButtonClicked();
    void onApply();
    void onReset();
    void onXAxisUseCustomLabelsToggled(bool checked);
    void onYAxisUseCustomLabelsToggled(bool checked);
    void onZAxisUseCustomLabelsToggled(bool checked);
    void onUseCustomBoundsToggled(bool checked);

private:
    void setupUI();
    void updateColorButton();
    void storeInitialValues();
    void restoreInitialValues();

    // Title Texts section
    QLineEdit* m_xTitleEdit;
    QLineEdit* m_yTitleEdit;
    QLineEdit* m_zTitleEdit;

    // Face Properties section
    QPushButton* m_gridColorButton;
    QColor m_currentColor;
    QCheckBox* m_showGridCheckBox;

    // X Axis Label Properties section
    QCheckBox* m_xAxisUseCustomLabelsCheckBox;
    ecvCustomLabelsWidget* m_xAxisCustomLabelsWidget;
    QGroupBox* m_xAxisGroup;

    // Y Axis Label Properties section
    QCheckBox* m_yAxisUseCustomLabelsCheckBox;
    ecvCustomLabelsWidget* m_yAxisCustomLabelsWidget;
    QGroupBox* m_yAxisGroup;

    // Z Axis Label Properties section
    QCheckBox* m_zAxisUseCustomLabelsCheckBox;
    ecvCustomLabelsWidget* m_zAxisCustomLabelsWidget;
    QGroupBox* m_zAxisGroup;

    // Bounds section
    QCheckBox* m_useCustomBoundsCheckBox;
    QWidget* m_customBoundsWidget;  // Container for bounds spinboxes
    QDoubleSpinBox* m_xMinSpinBox;
    QDoubleSpinBox* m_xMaxSpinBox;
    QDoubleSpinBox* m_yMinSpinBox;
    QDoubleSpinBox* m_yMaxSpinBox;
    QDoubleSpinBox* m_zMinSpinBox;
    QDoubleSpinBox* m_zMaxSpinBox;

    // Initial values for Reset functionality
    QString m_initialXTitle;
    QString m_initialYTitle;
    QString m_initialZTitle;
    QColor m_initialColor;
    bool m_initialShowGrid;
    bool m_initialXCustomLabels;
    bool m_initialYCustomLabels;
    bool m_initialZCustomLabels;
    QList<QPair<double, QString>> m_initialXLabels;
    QList<QPair<double, QString>> m_initialYLabels;
    QList<QPair<double, QString>> m_initialZLabels;
    bool m_initialCustomBounds;
    double m_initialXMin;
    double m_initialXMax;
    double m_initialYMin;
    double m_initialYMax;
    double m_initialZMin;
    double m_initialZMax;
};

#endif  // ECV_AXES_GRID_DIALOG_H
