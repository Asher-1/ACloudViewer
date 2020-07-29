#ifndef SIGNALBLOCKER_H
#define SIGNALBLOCKER_H

#include <QObject>

#include "../qPCL.h"

namespace VtkUtils
{

class QPCL_ENGINE_LIB_API SignalBlocker
{
public:
    explicit SignalBlocker(QObject* object = nullptr);
    ~SignalBlocker();

    void addObject(QObject* object);

private:
    QObjectList m_objectList;
};

} // namespace Utils
#endif // SIGNALBLOCKER_H
