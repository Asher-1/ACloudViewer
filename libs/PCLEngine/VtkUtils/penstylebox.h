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

#ifndef PEN_STYLE_BOX_H
#define PEN_STYLE_BOX_H


#include <QComboBox>

#include "../qPCL.h"

namespace Widgets
{

class QPCL_ENGINE_LIB_API PenStyleBox : public QComboBox
{
    Q_OBJECT

public:
    PenStyleBox(QWidget *parent = 0);
    void setStyle(const Qt::PenStyle& style);
    Qt::PenStyle style() const;

    static int styleIndex(const Qt::PenStyle& style);
    static Qt::PenStyle penStyle(int index);

private:
    static const Qt::PenStyle patterns[];
};

}
#endif
