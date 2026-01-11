// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>

#include "qPCL.h"

/**
 * @brief cvExpanderButton provides a frame with a toggle mode for collapsible
 * sections.
 *
 * This widget is based on ParaView's pqExpanderButton and is used to simulate
 * a toggle button for expanding/collapsing frames in an accordion style.
 *
 * When checked (expanded), it shows a minus icon (-).
 * When unchecked (collapsed), it shows a plus icon (+).
 */
class QPCL_ENGINE_LIB_API cvExpanderButton : public QFrame {
    Q_OBJECT

    Q_PROPERTY(QString text READ text WRITE setText)
    Q_PROPERTY(bool checked READ checked WRITE setChecked)

public:
    explicit cvExpanderButton(QWidget* parent = nullptr);
    ~cvExpanderButton() override;

public slots:
    /**
     * @brief Toggles the state of the checkable button.
     */
    void toggle();

    /**
     * @brief This property holds whether the button is checked.
     * By default, the button is unchecked (collapsed).
     */
    void setChecked(bool checked);
    bool checked() const { return m_checked; }

    /**
     * @brief This property holds the text shown on the button.
     */
    void setText(const QString& text);
    QString text() const;

signals:
    /**
     * @brief This signal is emitted whenever the button changes its state.
     * @param checked true if the button is checked (expanded), false if
     * unchecked (collapsed).
     */
    void toggled(bool checked);

protected:
    void mousePressEvent(QMouseEvent* evt) override;
    void mouseReleaseEvent(QMouseEvent* evt) override;

private:
    void setupUi();
    void updateIcon();

private:
    QHBoxLayout* m_layout;
    QLabel* m_iconLabel;
    QLabel* m_textLabel;

    bool m_checked;
    bool m_pressed;

    QPixmap m_checkedPixmap;    // Minus icon (expanded)
    QPixmap m_uncheckedPixmap;  // Plus icon (collapsed)
};
