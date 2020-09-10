#ifndef ACTOREXPORTER_H
#define ACTOREXPORTER_H

#include <QRunnable>
#include <QString>

#include "../qPCL.h"

class vtkActor;
namespace VtkUtils
{

class QPCL_ENGINE_LIB_API ActorExporter : public QRunnable
{
public:
    ActorExporter(vtkActor* actor, const QString& file);

    void run();

protected:
    vtkActor* m_actor = nullptr;
    QString m_exportFile;
};

} // namespace VtkUtils
#endif // ACTOREXPORTER_H
