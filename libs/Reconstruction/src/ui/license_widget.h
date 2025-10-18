// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtWidgets>

namespace colmap {

class LicenseWidget : public QTextEdit {
public:
    explicit LicenseWidget(QWidget* parent);

private:
    QString GetCOLMAPLicense() const;
    QString GetFLANNLicense() const;
    QString GetGraclusLicense() const;
    QString GetLSDLicense() const;
#ifdef PBA_ENABLED
    QString GetPBALicense() const;
#endif
    QString GetPoissonReconLicense() const;
    QString GetSiftGPULicense() const;
    QString GetSQLiteLicense() const;
    QString GetVLFeatLicense() const;
};

}  // namespace colmap
