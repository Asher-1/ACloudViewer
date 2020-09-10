#ifndef FILTERS_INTERFACE_H
#define FILTERS_INTERFACE_H

#include "../../qPCL.h"
#include <ecvWidgetsInterface.h>

// QT
#include <QMap>
#include <QWidget>

class QPCL_ENGINE_LIB_API FiltersWidgetInterface : public QObject, public DBLib::ecvWidgetsInterface
{
	Q_OBJECT

public:
	virtual ~FiltersWidgetInterface() {}

	virtual void initWidgets() override;

	//! Returns the default interface name
	virtual QString getInterfaceName() const override;

	virtual QList<QWidget*> getWidgts() const override;
	virtual QWidget* getWidgtByID(const DBLib::VTK_WIDGETS_TYPE widgetID) const override;
	virtual QStringList getWidgtsNames() const override;
	virtual const QString getWidgtNameByID(const DBLib::VTK_WIDGETS_TYPE widgetID) const override;

	virtual bool setInput(const ccHObject * obj, const DBLib::VTK_WIDGETS_TYPE widgetType) override;
	virtual ccHObject * getOutput(const DBLib::VTK_WIDGETS_TYPE widgetType) override;

	//! Called when the interface is unregistered
	/** Does nothing by default **/
	virtual void unregister() override;

protected:
	QMap<DBLib::VTK_WIDGETS_TYPE, QWidget*> m_widgetsMap;
	QMap<DBLib::VTK_WIDGETS_TYPE, QString> m_NamesMap;
};

#endif // FILTERS_INTERFACE_H
