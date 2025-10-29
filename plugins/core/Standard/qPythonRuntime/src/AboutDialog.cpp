// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AboutDialog.h"
#include "PythonInterpreter.h"
#include "Resources.h"

#include <ui_AboutDialog.h>

#undef slots
#include <pybind11/pybind11.h>

AboutDialog::AboutDialog(QWidget *parent) : QDialog(parent), m_dlg(new Ui_AboutDialog)
{
    m_dlg->setupUi(this);
    setWindowIcon(QIcon(ABOUT_ICON_PATH));

    connect(m_dlg->okBtn, &QPushButton::clicked, this, &QDialog::close);
    if (PythonInterpreter::IsInitialized())
    {
        const char *versionStr = Py_GetVersion();
        m_dlg->pythonVersionLabel->setText(QString(versionStr));
    }
    else
    {
        m_dlg->pythonVersionLabel->setText(QString("Unknown python version"));
    }
}

AboutDialog::~AboutDialog() noexcept
{
    delete m_dlg;
}
