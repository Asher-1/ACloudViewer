// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LasPlugin.h"

#include "LasIOFilter.h"
#include "LasVlr.h"

LasPlugin::LasPlugin(QObject* parent)
    : QObject(parent)
    , ccIOPluginInterface(":/CC/plugin/LAS-IO/info.json")
{
	qRegisterMetaType<LasVlr>();
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
	qRegisterMetaTypeStreamOperators<LasVlr>("LasVlr");
#endif

	QMetaType::registerConverter(&LasVlr::toString);
}

ccIOPluginInterface::FilterList LasPlugin::getFilters()
{
	return {
	    FileIOFilter::Shared(new LasIOFilter),
	};
}
