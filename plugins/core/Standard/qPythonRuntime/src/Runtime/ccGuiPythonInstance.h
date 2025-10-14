// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <FileIOFilter.h>

class ecvMainAppInterface;
class QMainWindow;

namespace pybind11
{
class args;
class kwargs;
class object;
} // namespace pybind11

/// This class provide methods that are made available Python scripts
/// kind of like C++ plugins have access to a `ecvMainAppInterface`.
/// Thus this class is mostly a ecvMainAppInterface with some accommodations
/// to handle the fact that it is made to interact with python.
class Q_DECL_EXPORT ccGuiPythonInstance final
{
  public:
    explicit ccGuiPythonInstance(ecvMainAppInterface *app) noexcept(false);

    QMainWindow *getMainWindow();

    bool haveSelection() const;

    bool haveOneSelection() const;

    const ccHObject::Container &getSelectedEntities() const;

    void setSelectedInDB(ccHObject *obj, bool selected);

    ccHObject *dbRootObject();

    void addToDB(pybind11::object &obj,
                 bool updateZoom = false,
                 bool autoExpandDBTree = true,
                 bool checkDimensions = false,
                 bool autoRedraw = true);

    void removeFromDB(pybind11::object &obj);

    void redrawAll(bool only2D = false);

    void refreshAll(bool only2D = false);

    void enableAll();

    void disableAll();

    void updateUI();

    void freezeUI(bool state);

    ccHObject *loadFile(const char *filename, FileIOFilter::LoadParameters &parameters);

    ecvMainAppInterface *app();

  private:
    ecvMainAppInterface *m_app;
};
