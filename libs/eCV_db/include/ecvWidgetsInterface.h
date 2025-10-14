// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_WIDGETS_INTERFACE_H
#define ECV_WIDGETS_INTERFACE_H

#include "eCV_db.h"

// QT
#include <QList>
#include <QSharedPointer>
#include <QString>
#include <QStringList>

class QWidget;
class ccHObject;
namespace DBLib {
//! Typical widget interface types
enum VTK_INTERFACES_TYPE { VTK_FILTERS, VTK_RENDER_SURFACE, VTK_SMALL_WIDGET };

//! Typical widget types
enum VTK_WIDGETS_TYPE {
    // filter widgets
    VTK_CLIP_WIDGET,
    VTK_SLICE_WIDGET,
    VTK_ISOSURFACE_WIDGET,
    VTK_THRESHOLD_WIDGET,
    VTK_STREAMLINE_WIDGET,
    VTK_GLYPH_WIDGET,
    VTK_PROB_WIDGET,
    VTK_SMOOTH_WIDGET,
    VTK_DECIMATEWIDGET_WIDGET,

    // render surface widgets
    VTK_SURFACE_WIDGET,

    // small widgets
    VTK_ANGLE_WIDGET,
    VTK_DISTANCE_WIDGET,
    VTK_CONTOUR_WIDGET,
};

class ECV_DB_LIB_API ecvWidgetsInterface {
public:
    //! Shared type
    typedef QSharedPointer<ecvWidgetsInterface> Shared;

public:
    static QWidget* LoadWidget(const VTK_WIDGETS_TYPE widgetType,
                               Shared filter);

    static QWidget* LoadWidget(const VTK_WIDGETS_TYPE widgetType);

    static bool SetInput(const ccHObject* obj,
                         const VTK_WIDGETS_TYPE widgetType);
    static ccHObject* GetOutput(const VTK_WIDGETS_TYPE widgetType);

    //! Init internal interfaces (should be called once)
    static void InitInternalInterfaces();

    //! Registers a new interface
    static void Register(Shared widgetInterface);

    //! Unregisters all interfaces
    /** Should be called at the end of the application
     **/
    static void UnregisterAll();

    //! Returns the interface corresponding to the given interface type
    static Shared GetWigetInterface(const VTK_WIDGETS_TYPE widgetID);

    //! Type of a Interface container
    typedef std::vector<ecvWidgetsInterface::Shared> InterfaceContainer;

    //! Returns the set of all registered interfaces
    static const InterfaceContainer& GetWigetInterfaces();

public:  // public interface (to be reimplemented by each widgets interface)
    //! Returns the widget(s) for this Interface
    /** E.g. 'filter widget
            \return list of widgets type
    **/
    virtual QStringList getWidgtsNames() const = 0;
    virtual const QString getWidgtNameByID(
            const VTK_WIDGETS_TYPE widgetID) const = 0;

    //! Returns the default interface name
    virtual QString getInterfaceName() const = 0;

    virtual void initWidgets() = 0;
    virtual QList<QWidget*> getWidgts() const = 0;
    virtual QWidget* getWidgtByID(const VTK_WIDGETS_TYPE widgetID) const = 0;

    //! Called when the interface is unregistered
    /** Does nothing by default **/
    virtual void unregister() = 0;

    virtual bool setInput(const ccHObject* obj,
                          const VTK_WIDGETS_TYPE widgetType) = 0;
    virtual ccHObject* getOutput(const VTK_WIDGETS_TYPE widgetType) = 0;
};

}  // namespace DBLib

#endif  // ECV_WIDGETS_INTERFACE_H
