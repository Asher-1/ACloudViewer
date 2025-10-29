// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>
#include <unordered_map>

namespace cloudViewer {

class OptionsWidget : public QWidget {
public:
    explicit OptionsWidget(QWidget* parent);

    void AddOptionRow(const std::string& label_text,
                      QWidget* widget,
                      void* option);
    void AddWidgetRow(const std::string& label_text, QWidget* widget);
    void AddLayoutRow(const std::string& label_text, QLayout* layout);

    QSpinBox* AddOptionInt(int* option,
                           const std::string& label_text,
                           const int min = 0,
                           const int max = static_cast<int>(1e7));
    QDoubleSpinBox* AddOptionDouble(double* option,
                                    const std::string& label_text,
                                    const double min = 0,
                                    const double max = 1e7,
                                    const double step = 0.01,
                                    const int decimals = 2);
    QDoubleSpinBox* AddOptionDoubleLog(double* option,
                                       const std::string& label_text,
                                       const double min = 0,
                                       const double max = 1e7,
                                       const double step = 0.01,
                                       const int decimals = 2);
    QCheckBox* AddOptionBool(bool* option, const std::string& label_text);
    QLineEdit* AddOptionText(std::string* option,
                             const std::string& label_text);
    QLineEdit* AddOptionFilePath(std::string* option,
                                 const std::string& label_text);
    QLineEdit* AddOptionDirPath(std::string* option,
                                const std::string& label_text);

    void AddSpacer();
    void AddSection(const std::string& title);

    void ReadOptions();
    void WriteOptions();

protected:
    void showEvent(QShowEvent* event);
    void closeEvent(QCloseEvent* event);
    void hideEvent(QHideEvent* event);

    void ShowOption(void* option);
    void HideOption(void* option);

    void ShowWidget(QWidget* option);
    void HideWidget(QWidget* option);

    void ShowLayout(QLayout* option);
    void HideLayout(QLayout* option);

    QGridLayout* grid_layout_;

    std::unordered_map<void*, std::pair<QLabel*, QWidget*>> option_rows_;
    std::unordered_map<QWidget*, std::pair<QLabel*, QWidget*>> widget_rows_;
    std::unordered_map<QLayout*, std::pair<QLabel*, QWidget*>> layout_rows_;

    std::vector<std::pair<QSpinBox*, int*>> options_int_;
    std::vector<std::pair<QDoubleSpinBox*, double*>> options_double_;
    std::vector<std::pair<QDoubleSpinBox*, double*>> options_double_log_;
    std::vector<std::pair<QCheckBox*, bool*>> options_bool_;
    std::vector<std::pair<QLineEdit*, std::string*>> options_text_;
    std::vector<std::pair<QLineEdit*, std::string*>> options_path_;
};

}  // namespace cloudViewer
