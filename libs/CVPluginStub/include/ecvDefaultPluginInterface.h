// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

#include "ecvPluginInterface.h"

class ccDefaultPluginData;

class ccDefaultPluginInterface : public ccPluginInterface {
public:
    virtual ~ccDefaultPluginInterface();

    virtual bool isCore() const override;

    virtual QString getName() const override;
    virtual QString getDescription() const override;

    virtual QIcon getIcon() const override;

    virtual ReferenceList getReferences() const override;
    virtual ContactList getAuthors() const override;
    virtual ContactList getMaintainers() const override;

protected:
    ccDefaultPluginInterface(const QString& resourcePath = QString());

private:
    void setIID(const QString& iid) override;
    const QString& IID() const override;

    ccDefaultPluginData* m_data;
};
