/*! \file
*  \brief Picture UI
*  \author Asher
*  \date 2013
*  \version 1.0
*  \copyright 2013 PERAGlobal Ltd. All rights reserved.
*
*  PenStyleBox
*/
/****************************************************************************
**
** Copyright (c) 2013 PERAGlobal Ltd. All rights reserved.
** All rights reserved.
**
****************************************************************************/

#ifndef COLORBUTTON_H
#define COLORBUTTON_H

#include "../qPCL.h"
#include "qtcolorpicker.h"

namespace Widgets {

class QPCL_ENGINE_LIB_API ColorPushButton : public QtColorPicker
{
    Q_OBJECT

public:
    ColorPushButton(QWidget *parent = 0);
    ~ColorPushButton();
    void setColor(const QColor& c){ setCurrentColor(c); }
    QColor color(){ return currentColor(); }

signals:
    void colorChanged();

private:
    Q_DISABLE_COPY(ColorPushButton)
};

}
#endif
