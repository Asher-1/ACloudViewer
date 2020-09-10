#ifndef MOVEPOINTINTERACTORSTYLE_H
#define MOVEPOINTINTERACTORSTYLE_H

#include "qPCL.h"

#include <QObject>

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API MovePointInteractorStyle : public QObject
{
    Q_OBJECT
public:
    explicit MovePointInteractorStyle(QObject *parent = 0);

signals:

public slots:
};

} // namespace VtkUtils
#endif // MOVEPOINTINTERACTORSTYLE_H
