// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QVector>

// qCC_io
#include <FileIOFilter.h>

#include "ecvDefaultPluginInterface.h"

//! I/O filter plugin interface
/** Version 1.3
 **/
class ccIOPluginInterface : public ccDefaultPluginInterface {
public:
    using FilterList = QVector<FileIOFilter::Shared>;

public:
    ccIOPluginInterface(const QString &resourcePath = QString())
        : ccDefaultPluginInterface(resourcePath) {}

    virtual ~ccIOPluginInterface() override = default;

    // inherited from ccPluginInterface
    virtual CC_PLUGIN_TYPE getType() const override {
        return ECV_IO_FILTER_PLUGIN;
    }

    //! Returns a list of I/O filter instances
    virtual FilterList getFilters() { return FilterList{}; }
};

Q_DECLARE_INTERFACE(ccIOPluginInterface,
                    "edf.rd.cloudviewer.ccIOFilterPluginInterface/1.3")
