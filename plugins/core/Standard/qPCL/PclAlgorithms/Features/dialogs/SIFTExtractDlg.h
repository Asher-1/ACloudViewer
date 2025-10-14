// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_SIFT_DLG_HEADER
#define Q_PCL_PLUGIN_SIFT_DLG_HEADER

#include <ui_SIFTExtractDlg.h>

// Qt
#include <QDialog>

// system
#include <vector>

class SIFTExtractDlg : public QDialog, public Ui::SIFTExtractDlg {
public:
    explicit SIFTExtractDlg(QWidget* parent = nullptr);

    void updateComboBox(const std::vector<std::string>& fields);
};

#endif  // Q_PCL_PLUGIN_SIFT_DLG_HEADER
