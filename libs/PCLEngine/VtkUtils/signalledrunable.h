#ifndef SIGNALLEDRUNNABLE_H
#define SIGNALLEDRUNNABLE_H

#include <QObject>
#include <QRunnable>

#include "../qPCL.h"

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API SignalledRunnable : public QObject, public QRunnable
{
    Q_OBJECT
public:
    SignalledRunnable();

signals:
    void finished();
};

} // namespace Utils
#endif // SIGNALLEDRUNABLE_H
