// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvExternalFactory.h"

//! Container singleton
static QSharedPointer<ccExternalFactory::Container> s_externalFactoryContainer(
        0);

ccExternalFactory::ccExternalFactory(QString factoryName)
    : m_factoryName(factoryName) {}

ccExternalFactory* ccExternalFactory::Container::getFactoryByName(
        const QString& factoryName) const {
    if (m_factories.contains(factoryName))
        return m_factories.value(factoryName);
    else
        return 0;
}

void ccExternalFactory::Container::addFactory(ccExternalFactory* factory) {
    if (!factory)  // do nothing
        return;

    QString name = factory->getFactoryName();

    m_factories[name] = factory;
}

ccExternalFactory::Container::Shared
ccExternalFactory::Container::GetUniqueInstance() {
    if (!s_externalFactoryContainer) {
        s_externalFactoryContainer =
                Container::Shared(new ccExternalFactory::Container());
    }
    return s_externalFactoryContainer;
}

void ccExternalFactory::Container::SetUniqueInstance(
        Container::Shared container) {
    s_externalFactoryContainer = container;
}
